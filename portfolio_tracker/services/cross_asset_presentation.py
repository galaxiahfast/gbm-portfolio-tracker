"""One compact display contract for UI and all PDF scopes."""


def cross_asset_rows(context):
    def number(value, digits=2):
        return "N/D" if value is None else f"{value:.{digits}f}"
    peer = context.get("peer", "activo par")
    ratio = context.get("ratio", {})
    rsis = context.get("rsi14", {})
    return [
        (f"Correlación con {peer}", number(context.get("correlation"))),
        ("Retornos diarios / corte", f"{context.get('return_samples', 0)} retornos · {context.get('daily_as_of', 'N/D')}"),
        ("Ratio SMCI/NVDA · diario cerrado", number(ratio.get("value"), 4)),
        ("Media 50 sesiones / desviación", f"{number(ratio.get('mean50'), 4)} / {number(ratio.get('deviation_pct'))}%"),
        ("RSI14 diario · SMCI / NVDA", f"{number(rsis.get('SMCI'), 1)} / {number(rsis.get('NVDA'), 1)}"),
        ("Ajuste de score · propuesto / aplicado", f"{context.get('proposed_impact', 0):+.1f} / {context.get('applied_impact', 0):+.1f} puntos"),
        ("Corte intradía cerrado (UTC)", context.get("intraday_as_of") or "N/D · sin ajuste intradía"),
    ]
