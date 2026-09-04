"""Descriptive global evidence, stratified by version and event (never mixed)."""
from collections import defaultdict
from .zone_forward import validation_data


def global_validation_summary(rows):
    groups = defaultdict(list)
    for row in validation_data(rows):
        for symbol in (row["symbol"], "GLOBAL"):
            groups[(row["model_version_hash"], row["event"], symbol)].append(row)
    results = []
    for (version, event, symbol), members in sorted(groups.items()):
        sessions = defaultdict(list)
        for row in members:
            sessions[row["session_date"]].append(row["brier"])
        results.append({"Versión": version, "Evento": event, "Emisora": symbol,
                        "Evaluadas": len(members), "Ocurrieron": sum(r["actual"] for r in members),
                        "Frecuencia observada": sum(r["actual"] for r in members) / len(members),
                        "Sesiones": len(sessions),
                        "Brier · peso igual por sesión": sum(sum(v)/len(v) for v in sessions.values())/len(sessions)})
    return results
