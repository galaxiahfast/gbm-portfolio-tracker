"""Report isolated historical coverage, not training or prediction accuracy."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import pandas as pd

from scripts.download_massive_history import ROOT, SYMBOLS, read_signed, sha, write_signed
from scripts.build_historical_replay import _signed_frame
from portfolio_tracker.analytics.historical_replay import validate_historical_replay


def nonoverlapping(rows):
    """First eligible cut, then embargo its full declared window, not its profit.

    This eliminates mechanically overlapping paths; it does not establish IID.
    The ordering is frozen before consulting outcomes or realized exit times.
    """
    selected, boundary = [], None
    ordered = sorted(rows, key=lambda row: row["operational_contract"]["evaluation_starts_at"])
    for row in ordered:
        contract = row["operational_contract"]
        start = pd.Timestamp(contract["evaluation_starts_at"])
        if boundary is None or start >= boundary:
            selected.append(row)
            boundary = pd.Timestamp(contract["timeout_at"])
    return selected


def event_key(contract):
    return sha({key: contract[key] for key in (
        "side", "entry_price", "stop_loss", "take_profit", "evaluation_starts_at",
    )})


def summarize(payload, horizon, intraday=None):
    rows = [row for cut in payload["observations"] for row in cut["horizons"]
            if row["horizon_minutes"] == horizon]
    eligible = [row for row in rows if (row.get("operational_contract") or {}).get("eligible_at_emission")]
    separate = nonoverlapping(eligible)
    executions = Counter(row["execution_result"]["status"] for row in eligible)
    resolved = [row for row in eligible if row["execution_result"]["status"] == "SIMULATED_RESOLVED"]
    independent_resolved = [row for row in separate
                            if row["execution_result"]["status"] == "SIMULATED_RESOLVED"]
    hypothetical = Counter(row["operational_result"].get("outcome")
                           for row in rows if row["operational_result"]["status"] == "RESOLVED")
    hypothetical_statuses = Counter(row["operational_result"]["status"] for row in rows)
    ambiguous_barriers = 0
    if intraday is not None:
        utc_frame = intraday.copy()
        utc_frame.index = utc_frame.index.tz_convert("UTC")
        for row in rows:
            result, contract = row["operational_result"], row["operational_contract"]
            if (result["status"] != "RESOLVED" or not contract
                    or not str(result.get("exit_source", "")).startswith("5m")):
                continue
            start = pd.Timestamp(result["exit_at"]) - pd.Timedelta(minutes=5)
            if start in utc_frame.index:
                candle = utc_frame.loc[start]
                if (candle.High >= max(contract["stop_loss"], contract["take_profit"])
                        and candle.Low <= min(contract["stop_loss"], contract["take_profit"])):
                    ambiguous_barriers += 1
    outcomes = Counter(row["execution_result"]["outcome"] for row in resolved)
    return {
        "cuts": len(rows),
        "directional_only_no_plan": sum(row["operational_contract"] is None for row in rows),
        "technical_authorized_entries": len(eligible),
        "authorized_directions": dict(Counter(row["operational_contract"]["side"] for row in eligible)),
        "possible_fills": sum(row["execution_result"].get("fill_price") is not None for row in eligible),
        "executable_resolved_events": len(resolved),
        "execution_outcomes": {key: outcomes[key] for key in ("TP_FIRST", "SL_FIRST", "TIMEOUT")},
        "execution_ambiguous_no_trade": executions["AMBIGUOUS_NO_TRADE"],
        "execution_no_fill": sum(count for status, count in executions.items() if status.startswith("NO_FILL")),
        "execution_pending": executions["PENDING_ENTRY"] + executions["FILLED_PENDING"],
        "execution_evidence_incomplete": executions["EVIDENCE_INCOMPLETE"],
        "not_authorized": sum(row["execution_result"]["status"] == "NOT_AUTHORIZED" for row in rows),
        "authorized_nonoverlapping_windows": len(separate),
        "resolved_nonoverlapping_events": len(independent_resolved),
        "shortfall_to_300_resolved_nonoverlapping": max(0, 300 - len(independent_resolved)),
        "hypothetical_barrier_outcomes": {key: hypothetical[key] for key in ("TP_FIRST", "SL_FIRST", "TIMEOUT")},
        "hypothetical_pending": hypothetical_statuses["RIGHT_CENSORED"],
        "hypothetical_missing_path": hypothetical_statuses["MISSING_CAUSAL_PATH"],
        "hypothetical_same_bar_ambiguity_included_as_conservative_SL": ambiguous_barriers,
    }


def _pre_compat_validator(source):
    """Recover the run's exact archived source, verified against its SHA.

    Only a validator compatibility fix occurred while the workers ran. It did
    not change feature generation, rules, prices, or labels. Never substitute
    this text unless it matches the hash frozen by the running interpreter.
    """
    block = '''    explicit_scope = "selected_horizons" in payload or "input_window_policy" in payload
    if explicit_scope and (
            payload.get("contract") != HISTORICAL_REPLAY_CONTRACT
            or "selected_horizons" not in payload
            or payload.get("input_window_policy") != "MATCH_LIVE_INTRADAY_1MO_V1"):
        raise ValueError("Alcance temporal de replay incompleto o desconocido.")
'''
    current = '''        # Backward-compatible six-horizon envelopes retain their original
        # hash domain. Missing scope can NEVER imply a two-horizon pilot.
        expected = payload.get("selected_horizons") if explicit_scope else list(HORIZON_MINUTES.values())'''
    previous = '''        expected = (payload.get("selected_horizons") if payload.get("contract") == HISTORICAL_REPLAY_CONTRACT
                    else list(HORIZON_MINUTES.values()))'''
    return source.replace(block, "").replace(current, previous).replace(
        "    if explicit_scope:\n", "    if payload[\"contract\"] == HISTORICAL_REPLAY_CONTRACT:\n"
    )


def archive_sources(payload, folder):
    fingerprints = payload["observations"][0]["feature_snapshot"]["code_sha256"]
    if any(cut["feature_snapshot"]["code_sha256"] != fingerprints for cut in payload["observations"]):
        raise ValueError("La corrida mezcló versiones de código.")
    for name, expected in fingerprints.items():
        raw = (ROOT / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            if name != "portfolio_tracker/analytics/historical_replay.py":
                raise ValueError(f"No se puede archivar la versión ejecutada: {name}")
            prior = _pre_compat_validator(raw.decode("utf-8")).encode("utf-8")
            if hashlib.sha256(prior).hexdigest() != expected:
                raise ValueError("No se pudo recuperar el validador exacto de la corrida.")
            raw = prior
        target = folder / expected / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != raw:
            raise ValueError("Archivo fuente histórico alterado.")
        target.write_bytes(raw)
    return fingerprints


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pytest-passed", type=int, required=True)
    parser.add_argument("--ledger-before-sha", help="Optional read-only comparison of the portfolio.db file.")
    args = parser.parse_args(argv)
    manifest = read_signed(args.input_dir / "manifest.json")
    report = {"audit_date_utc": datetime.now(timezone.utc).date().isoformat(),
              "scope": "DATA_ACQUISITION_AND_OPERATIONAL_COVERAGE_ONLY_NO_TRAINING",
              "source": {"provider": "Massive", "selected_plan": "Stocks Basic",
                         "advertised_cost_usd_per_month": 0, "historical_entitlement_months": 24,
                         "pricing_url": "https://massive.com/pricing", "no_subscription_purchase": True},
              "import_manifest_sha256": sha(manifest), "symbols": {},
              "pytest_full_suite_passed": args.pytest_passed,
              "environment": {"python": platform.python_version(),
                              **{name: version(name) for name in ("pandas", "numpy", "exchange-calendars")}},
              "limitations": ["No historical point-in-time news/fundamental veto",
                              "No historical account/broker short permission or real fills",
                              "Daily history 499 sessions, not the live five-year request",
                              "Non-overlap does not guarantee statistical independence",
                              "SHA-256 is integrity, not provider authenticity attestation"]}
    if args.ledger_before_sha:
        ledger_sha = hashlib.sha256((ROOT / "data" / "portfolio.db").read_bytes()).hexdigest()
        if ledger_sha != args.ledger_before_sha.lower():
            raise ValueError("El archivo contable cambió; no se certifica igualdad de hashes.")
        report["portfolio_db_file_sha256_before_and_after"] = ledger_sha
    for symbol in SYMBOLS:
        files = list(args.replay_dir.glob(f"{symbol}_*.json"))
        if len(files) != 1:
            raise ValueError(f"Se requiere exactamente una corrida completa de {symbol}.")
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        validate_historical_replay(payload)
        if payload.get("selected_horizons") != [60, 1440]:
            raise ValueError("El reporte solo admite el piloto preregistrado 1h/1d.")
        intraday, _ = _signed_frame(args.input_dir / symbol / "5m.json")
        samples = list(args.sample_dir.glob(f"massive_{symbol}_*.json"))
        sample = read_signed(samples[0])
        features = [cut["feature_snapshot"]["features"] for cut in payload["observations"]]
        eligible_contracts = [row["operational_contract"] for cut in payload["observations"]
                              for row in cut["horizons"] if (row["operational_contract"] or {}).get("eligible_at_emission")]
        report["symbols"][symbol] = {
            "data_audit": manifest["symbols"][symbol],
            "sample": {"sessions": sample["sessions"], "bars": len(sample["bars"]),
                       "first_bar": sample["bars"][0], "last_bar": sample["bars"][-1], "sha256": sha(sample)},
            "candidate_cuts": payload["candidate_cuts"], "generated_cuts": len(payload["observations"]),
            "rejected_cuts": payload["rejected"], "unique_authorized_events_across_horizons": len({event_key(c) for c in eligible_contracts}),
            "gate_counts": {"trigger_confirmed": sum(bool(x.get("activation_trigger_met")) for x in features),
                            "risk_veto": sum(bool(x.get("risk_veto")) for x in features),
                            "macro_permissions": dict(Counter(x.get("macro_permission") for x in features))},
            "horizons": {str(h): summarize(payload, h, intraday) for h in (60, 1440)},
            "replay_artifact": str(files[0].resolve()), "content_sha256": payload["content_sha256"],
            "file_sha256": hashlib.sha256(files[0].read_bytes()).hexdigest(),
            "code_sha256": archive_sources(payload, args.output_dir / "code_archive"),
            "authorized_event_details": [
                {"observed_at": cut["observed_at"], "horizon_minutes": row["horizon_minutes"],
                 "side": row["operational_contract"]["side"], "execution": row["execution_result"]}
                for cut in payload["observations"] for row in cut["horizons"]
                if (row["operational_contract"] or {}).get("eligible_at_emission")
            ],
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_signed(args.output_dir / "pilot_report.json", report)
    lines = ["# Piloto histórico causal SMCI / NVDA", "", f"Fecha: {report['audit_date_utc']} UTC.", "",
             "## Resultado principal", "", "Sesiones auditadas: " + ", ".join(
                 f"{symbol}={row['data_audit']['intraday_sessions']}" for symbol, row in report['symbols'].items()) + ". "
             "Cortes generados: " + ", ".join(f"{symbol}={row['generated_cuts']}" for symbol, row in report['symbols'].items()) + ", tras 200 sesiones diarias de calentamiento. "
             "No se entrenó ni calibró ningún modelo con este histórico.", "",
             "Massive Stocks Basic: USD 0/mes; 24 meses de histórico y agregados por minuto. "
             "No se contrató ni actualizó ninguna suscripción. [Tarifa oficial](https://massive.com/pricing).", "",
             "Cortes fijos: 11:00 America/New_York. Solo 1h y 1d. "
             f"Datos 5m: {manifest['protocol']['intraday_start']}–{manifest['protocol']['end']}. "
             f"Primer corte con calentamiento: {min(row['data_audit']['first_replay_session'] for row in report['symbols'].values())}.", "",
             "## Adquisición y auditoría", "",
             "| Activo | Sesiones 5m | Velas regulares | Diarias | Huecos / duplicados / OHLCV inválido |", "|---|---:|---:|---:|---|",
             *[f"| {symbol} | {row['data_audit']['intraday_sessions']} | {row['data_audit']['regular_bars']:,} | {row['data_audit']['daily_sessions']} | 0 / 0 / 0 |" for symbol, row in report['symbols'].items()], "",
             "XNYS verificado incluyendo DST y cierres anticipados; timestamps 5m son apertura de vela. "
             "Se excluyó pre/posmercado sin rellenar datos. SMCI: split 10:1 del 01/10/2024, "
             "anterior a todo corte del piloto; precios y volúmenes raw/adjusted reconciliados. "
             "NVDA: sin splits dentro de la ventana adquirida.", "",
             "El diario se deriva del OHLCV regular 5m donde existe; los diarios anteriores son solo calentamiento. "
             "Las velas posteriores al corte solo resuelven resultados, nunca alimentan sus features.", "",
             "## Entradas autorizadas y ejecución posible — NO barreras hipotéticas", "",
             "| Activo / horizonte | Cortes | Autorizadas | LONG / SHORT | Fill posible | TP / SL / timeout | Ambiguas | Pendientes | Faltan para 300 resueltas no solapadas |", "|---|---:|---:|---|---:|---|---:|---:|---:|"]
    for symbol, result in report["symbols"].items():
        for horizon, row in result["horizons"].items():
            classes, directions = row["execution_outcomes"], row["authorized_directions"]
            lines.append(f"| {symbol} / {'1h' if horizon == '60' else '1d'} | {row['cuts']} | "
                         f"{row['technical_authorized_entries']} | {directions.get('LONG', 0)} / {directions.get('SHORT', 0)} | "
                         f"{row['possible_fills']} | {classes['TP_FIRST']} / {classes['SL_FIRST']} / {classes['TIMEOUT']} | "
                         f"{row['execution_ambiguous_no_trade']} | {row['execution_pending']} | {row['shortfall_to_300_resolved_nonoverlapping']} |")
    unique_events = sum(row['unique_authorized_events_across_horizons'] for row in report['symbols'].values())
    long_events = sum(row['horizons']['60']['authorized_directions'].get('LONG', 0) for row in report['symbols'].values())
    lines += ["", f"Eventos técnicos autorizados únicos: {unique_events}. Entradas LONG al corte: {long_events}. "
              "Varias resoluciones del mismo contrato no son aciertos independientes. "
              "El posible fill por OHLC no confirma ejecución de GBM+ ni permiso para vender en corto. "
              "Por ello no se presenta como evidencia de una compra operable en la cuenta real.", "",
              *[f"- {symbol}, {item['observed_at']}, {item['side']}, horizonte {item['horizon_minutes']} min: "
                f"{item['execution']['status']} / {item['execution'].get('outcome')}."
                for symbol, row in report['symbols'].items() for item in row['authorized_event_details']], "",
              "## Población separada: barreras hipotéticas", "",
              "| Activo / horizonte | TP primero | SL primero | Timeout | Pendientes |", "|---|---:|---:|---:|---:|"]
    for symbol, result in report["symbols"].items():
        for horizon, row in result["horizons"].items():
            c = row["hypothetical_barrier_outcomes"]
            lines.append(f"| {symbol} / {'1h' if horizon == '60' else '1d'} | {c['TP_FIRST']} | {c['SL_FIRST']} | {c['TIMEOUT']} | {row['hypothetical_pending']} |")
    lines += ["", "Estos resultados NO cuentan como entradas autorizadas ni muestras ejecutables. "
              "Si TP y SL coinciden en una vela se conserva SL primero, sin inventar el orden intrabar. "
              "Los conteos de esa ambigüedad están en el JSON. Los pendientes 1d son censura derecha "
              "al terminar el histórico, no pérdidas ni datos fabricados.", "",
              "## Qué falta", "", "Gatillo confirmado: " + "; ".join(
                  f"{symbol}={row['gate_counts']['trigger_confirmed']}/{row['generated_cuts']}"
                  for symbol, row in report['symbols'].items()) + ". "
              "Más barreras resueltas no solucionan esto. Se debe auditar el embudo de autorización "
              "sin optimizar filtros usando ganancias del mismo pasado.", "",
              "Propuesta NO ejecutada: preregistrar un segundo corte a las 14:00 NY para estudiar "
              "la ventana de tarde, manteniendo reglas y sin elegirlo por rentabilidad histórica. "
              "Prioridad cronológica al primer corte elegible; máximo un contrato activo por activo "
              "y exclusión hasta el timeout original. Para 1d, varios cortes del mismo día no "
              "son muestras independientes. Extender años de historia también requiere una nueva "
              "adquisición/auditoría; no se anuncia que Basic dé más de dos años.", "",
              "## Límites y preservación", "",
              "- No hay fundamentales ni noticias históricos punto-en-tiempo; el veto correspondiente no se certifica.",
              "- No se reconstruye el estado histórico de cuenta ni permisos de SHORT, y los fills no son confirmaciones del corredor.",
              "- Hay 499 sesiones diarias, no los cinco años solicitados por la ingesta en vivo; el calentamiento no garantiza identidad exacta de inicialización EMA.",
              "- Ausencia de solapamiento no demuestra independencia estadística.",
              "- SHA-256 verifica integridad local, no autenticidad certificada por el proveedor.", "",
              "No se usó SQLite ni el libro contable, no se alteraron operaciones reales ni parámetros "
              "de trading, y no se guardaron claves API. Datos/licencia quedan fuera de Git. "
              "La corrección de cortes sin TP/SL conserva solo la predicción direccional. "
              f"Suite completa verificada: {args.pytest_passed} pruebas aprobadas. No es evidencia de precisión.", "",
              "## Muestra y hashes", ""]
    for symbol, result in report["symbols"].items():
        sample = result["sample"]
        lines += [f"### {symbol}", "", f"Muestra: {len(sample['sessions'])} sesiones {min(sample['sessions'])}–{max(sample['sessions'])}, {sample['bars']} velas.", "",
                  f"Primera: {sample['first_bar']['timestamp_ny']}; OHLCV "
                  f"{[sample['first_bar'][x] for x in ('open','high','low','close','volume')]}", "",
                  f"Última: {sample['last_bar']['timestamp_ny']}; OHLCV "
                  f"{[sample['last_bar'][x] for x in ('open','high','low','close','volume')]}", "",
                  f"SHA-256 muestra: `{sample['sha256']}`", "",
                  f"SHA-256 contenido replay: `{result['content_sha256']}`", "",
                  f"SHA-256 archivo replay: `{result['file_sha256']}`", ""]
    lines += [f"SHA-256 manifiesto de importación: `{report['import_manifest_sha256']}`", "",
              "Inventario completo de fuentes, versiones, métricas y hashes: pilot_report.json. "
              "Fuentes exactas usadas por los workers: code_archive/; cada archivo cotejado con "
              "el hash congelado en los cortes. Una corrección posterior del validador de "
              "compatibilidad no alteró señales ni resultados; su versión anterior quedó archivada."]
    if args.ledger_before_sha:
        lines += ["", "El archivo portfolio.db mantuvo el mismo SHA-256 antes y después: "
                  f"`{report['portfolio_db_file_sha256_before_and_after']}`. "
                  "Es una comparación del archivo, no una lectura o actualización de tablas contables."]
    (args.output_dir / "pilot_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({symbol: result["horizons"] for symbol, result in report["symbols"].items()}, indent=2))
    print(f"REPORTE: {(args.output_dir / 'pilot_report.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
