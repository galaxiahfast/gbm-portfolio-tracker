"""Read-only validation UI, except explicit/cached resolution of analytic logs."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..db import Database
from ..repository import PortfolioRepository
from ..services.forward_market import resolution_frames
from ..services.market_data import download_daily_history
from ..services.zone_forward import validation_data


@st.cache_data(ttl=900, max_entries=4, show_spinner=False)
def catch_up(database_path):
    repository = PortfolioRepository(Database(database_path))
    repository.ensure_zone_forward_schema()
    return repository.resolve_predictions(resolution_frames)


@st.cache_data(ttl=21600, max_entries=8, show_spinner=False)
def load_context(symbol):
    _, metadata = download_daily_history(symbol)
    return {key: value for key, value in metadata.items() if key != "frame"}


def validation_panel(repository, resolution_status):
    st.subheader("VALIDACIÓN · Evidencia prospectiva")
    st.caption("PRELIMINAR · Frecuencias observadas, no calibración OOS demostrada. "
               "Se registran los seis niveles disponibles durante la sesión; N/D no se convierte en 0%.")
    if st.button("Resolver cierres pendientes", key="forward_resolve"):
        catch_up.clear()
        st.rerun()
    for error in resolution_status.get("errors", []):
        st.warning(error)
    rows = repository.zone_predictions()
    bad = sum(not row["integrity_ok"] for row in rows)
    if bad:
        st.error(f"{bad} registros con integridad inválida: excluidos de todas las métricas.")
    valid = [row for row in rows if row["integrity_ok"]]
    symbols = sorted({row["symbol"] for row in valid}) or ["SMCI"]
    symbol = st.selectbox("Emisora registrada", symbols, key="forward_symbol")
    versions = sorted({row["model_version_hash"] for row in valid if row["symbol"] == symbol})
    model = st.selectbox("Versión del modelo", versions, format_func=lambda value: value[:16],
                         key="forward_model") if versions else None
    selected = [r for r in rows if r["symbol"] == symbol and r["model_version_hash"] == model]
    zone = st.selectbox("Nivel evaluado", ["Todos", "ENTRY1", "ENTRY2", "ENTRY3", "TP1", "TP2", "R3"],
                        key="forward_zone")
    if zone != "Todos":
        selected = [r for r in selected if r["zone_key"] == zone]
    with st.container(horizontal=True):
        st.metric("Predicciones registradas", len(selected), border=True)
        st.metric("Sesiones distintas", len({r["session_date"] for r in selected}), border=True)
        st.metric("Pendientes de resolver", sum(r["resolved_at"] is None for r in selected), border=True)
    st.caption("Cohorte principal: primera emisión por zona y sesión, elegida antes de conocer el resultado. "
               "Las seis zonas y las recargas no son observaciones independientes. "
               "Brier menor es mejor; por sí solo no demuestra que un 71% esté calibrado.")
    scored = validation_data(selected)
    if not scored:
        st.info("Aún no hay resultados prospectivos evaluables. No se rellenará este panel con pruebas sintéticas ni backtests retrospectivos.")
    else:
        frame = pd.DataFrame(scored)
        event = st.selectbox("Evento evaluado", ["Toque", "Cierre"], key="forward_event")
        frame = frame.loc[frame.event == event]
        if not frame.empty:
            # Each session receives equal weight, not each intraday refresh.
            daily = frame.groupby("session_date").agg(Brier=("brier", "mean"), Evaluadas=("actual", "size"),
                                                      Alcanzadas=("actual", "sum"))
            daily["Brier acumulado"] = daily.Brier.expanding().mean()
            st.metric("Brier acumulado · peso igual por sesión", f"{daily['Brier acumulado'].iloc[-1]:.4f}")
            st.line_chart(daily[["Brier acumulado"]])
            bins = frame.groupby("bin").agg(Emitida=("prediction", "mean"), Observada=("actual", "mean"), Alcanzadas=("actual", "sum"),
                                            N=("actual", "size"), Sesiones=("session_date", "nunique")).reset_index()
            figure = go.Figure()
            figure.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Referencia ideal",
                               line=dict(color="#888888", dash="dash"))
            figure.add_scatter(x=bins.Emitida, y=bins.Observada, mode="lines+markers",
                               name="Frecuencia observada", line=dict(color="#FFFFFF"),
                               customdata=bins[["N", "Sesiones"]],
                               hovertemplate="Emitida %{x:.1%}<br>Observada %{y:.1%}<br>N=%{customdata[0]} · sesiones=%{customdata[1]}<extra></extra>")
            figure.update_layout(height=320, xaxis=dict(title="Pronóstico medio", range=[0, 1], tickformat=".0%"),
                                 yaxis=dict(title="Frecuencia real", range=[0, 1], tickformat=".0%"),
                                 paper_bgcolor="#111111", plot_bgcolor="#111111", font_color="#FFFFFF")
            st.plotly_chart(figure)
            st.caption("Intervalos fijos de 10 puntos porcentuales. Una observación cercana a 71% "
                       "se agrupa en 70–80%; no demuestra exactitud del 71% puntual.")
            st.dataframe(bins.rename(columns={"Emitida": "Pronóstico medio", "Observada": "Frecuencia real"}),
                         hide_index=True)
            last_day = daily.index[-1]
            st.markdown(f"**Última sesión evaluada: {last_day}**")
            recent = frame.loc[frame.session_date == last_day]
            st.dataframe(recent[["zone_key", "zone_low", "zone_high", "timestamp_prediction",
                                 "prediction", "actual", "actual_close_price", "brier"]].rename(columns={
                "zone_key": "Zona", "zone_low": "Desde", "zone_high": "Hasta",
                "timestamp_prediction": "Emitida UTC", "prediction": "Pronóstico",
                "actual": "Evento ocurrió (1/0)", "actual_close_price": "Cierre real", "brier": "Brier"}),
                column_config={"Pronóstico": st.column_config.NumberColumn(format="percent")},
                hide_index=True)
    with st.expander("Registro completo y exclusiones"):
        if selected:
            st.dataframe(pd.DataFrame(selected)[["id", "session_date", "zone_key", "timestamp_prediction",
                "predicted_touch_probability", "predicted_close_probability", "actual_touch_occurred",
                "actual_close_price", "resolution_note", "resolved_at", "forecast_sha256"]],
                hide_index=True)
        else:
            st.write("Sin registros para esta versión.")
    with st.expander("Contexto diario / semanal / mensual desde 2024", on_change="rerun") as context:
        if context.open:
            try:
                with st.spinner("Verificando histórico diario..."):
                    metadata = load_context(symbol)
                st.caption(f"{metadata['bars']} velas · {metadata['first_session']} → {metadata['last_session']} · "
                           f"Descarga UTC: {metadata['fetched_at']} · No son predicciones históricas.")
                st.dataframe(pd.DataFrame(metadata["context"]).T)
                st.caption("EMA sin suficientes periodos: N/D. Este contexto no recalibra automáticamente el modelo de zonas.")
            except (ValueError, RuntimeError, OSError) as exc:
                st.warning(f"Histórico no disponible: {exc}")
