"""Gráficas financieras coherentes con el sistema visual monocromático."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
import streamlit as st


CHART_COLORS = (
    "#F2F2F2",
    "#CFCFCF",
    "#A8A8A8",
    "#858585",
    "#686868",
    "#B9B9B9",
    "#767676",
)
LINE_DASHES = ("solid", "dash", "dot", "dashdot")


def _numeric_frame(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.Series):
        frame = data.to_frame(name=data.name or "Valor")
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
    else:
        frame = pd.DataFrame(data)
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(how="all")


def premium_line_chart(
    data: Any,
    *,
    height: int = 300,
    key: str | None = None,
    colors: Sequence[str] = CHART_COLORS,
) -> None:
    """Renderiza líneas suavizadas con tooltip unificado y fondo transparente.

    El dataframe recibido no se altera. Los histogramas MACD conservan barras y
    las líneas de umbral se mantienen rectas para no distorsionar su lectura.
    """

    import plotly.graph_objects as go

    frame = _numeric_frame(data)
    if frame.empty:
        st.info("No hay observaciones suficientes para mostrar esta gráfica.")
        return

    figure = go.Figure()
    line_columns = [
        column for column in frame.columns if "histograma" not in str(column).lower()
    ]
    single_line = len(line_columns) == 1
    for index, column in enumerate(frame.columns):
        series = frame[column]
        label = str(column)
        color = colors[index % len(colors)]
        if "histograma" in label.lower():
            bar_colors = ["#D8D8D8" if value >= 0 else "#666666" for value in series.fillna(0)]
            figure.add_trace(
                go.Bar(
                    x=frame.index,
                    y=series,
                    name=label,
                    marker={"color": bar_colors, "opacity": 0.44, "line": {"width": 0}},
                    hovertemplate=f"{label}: %{{y:,.3f}}<extra></extra>",
                )
            )
            continue
        threshold = label.lower() in {"sobrecompra", "sobreventa"}
        figure.add_trace(
            go.Scatter(
                x=frame.index,
                y=series,
                name=label,
                mode="lines",
                connectgaps=False,
                line={
                    "color": color,
                    "width": 1.25 if threshold else 2.15,
                    "shape": "linear" if threshold else "spline",
                    "smoothing": 0 if threshold else 0.42,
                    "dash": "dot" if threshold else LINE_DASHES[index % len(LINE_DASHES)],
                },
                fill=None,
                hovertemplate=f"{label}: %{{y:,.3f}}<extra></extra>",
            )
        )

    finite_values = frame.to_numpy(dtype=float)
    finite_values = finite_values[pd.notna(finite_values)]
    yaxis: dict[str, Any] = {
        "showgrid": True,
        "gridcolor": "rgba(170, 170, 170, 0.10)",
        "gridwidth": 1,
        "zeroline": False,
        "fixedrange": False,
        "tickfont": {"color": "#858585", "size": 11},
    }
    if len(finite_values):
        low, high = float(finite_values.min()), float(finite_values.max())
        span = max(high - low, abs(high) * 0.03, 1e-9)
        yaxis["range"] = [low - span * 0.10, high + span * 0.12]

    figure.update_layout(
        height=height,
        margin={"l": 8, "r": 10, "t": 12, "b": 16},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": "#C8C8C8", "size": 11},
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "rgba(10, 10, 10, .96)",
            "bordercolor": "rgba(190, 190, 190, .28)",
            "font": {"color": "#F5F5F5", "family": "Inter, sans-serif"},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "x": 0,
            "font": {"color": "#A0A0A0", "size": 10},
        },
        xaxis={
            "showgrid": False,
            "zeroline": False,
            "rangeslider": {"visible": False},
            "tickfont": {"color": "#777777", "size": 10},
        },
        yaxis=yaxis,
        bargap=0.12,
    )
    st.plotly_chart(
        figure,
        width="stretch",
        key=key,
        config={
            "displaylogo": False,
            "displayModeBar": False,
            "scrollZoom": False,
            "responsive": True,
        },
    )


def premium_bar_chart(
    data: Any,
    *,
    height: int = 300,
    key: str | None = None,
) -> None:
    """Renderiza una distribución compacta con barras monocromáticas."""

    import plotly.graph_objects as go

    frame = _numeric_frame(data)
    if frame.empty:
        st.info("No hay datos suficientes para mostrar la distribución.")
        return
    column = frame.columns[0]
    figure = go.Figure(
        go.Bar(
            x=[str(item) for item in frame.index],
            y=frame[column],
            marker={
                "color": "#BDBDBD",
                "line": {"color": "rgba(210,210,210,.30)", "width": 1},
            },
            hovertemplate="%{x}: $%{y:,.2f}<extra></extra>",
        )
    )
    figure.update_layout(
        height=height,
        margin={"l": 8, "r": 8, "t": 12, "b": 14},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font={"family": "Inter, sans-serif", "color": "#A0A0A0"},
        xaxis={"showgrid": False, "tickfont": {"color": "#858585"}},
        yaxis={"showgrid": True, "gridcolor": "rgba(170,170,170,.10)", "zeroline": False},
    )
    st.plotly_chart(
        figure,
        width="stretch",
        key=key,
        config={"displaylogo": False, "displayModeBar": False, "responsive": True},
    )
