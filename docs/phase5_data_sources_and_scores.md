# Fase 5: procedencia de datos y scores (05-10-2026)

| Componente | Fuente verificada | Uso | Límite de comparabilidad |
|---|---|---|---|
| Archivo prospectivo 5 cortes | Alpaca **IEX**, OHLCV 5m y 1D `adjustment=split`; Finnhub para titulares con recepción propia | Snapshots firmados; `data_source=ALPACA_IEX_SPLIT_ADJUSTED` | IEX no representa todo el volumen consolidado. No mezclar con otra fuente dentro de un reporte/cohorte. |
| App en vivo | Yahoo Finance vía `yfinance`; si falla, último corte local SHA-256 (`portfolio_tracker/services/predictor_market_data.py`) | Visualización y análisis interactivo | El respaldo local es un corte antiguo identificado, no una cotización Yahoo en vivo. No combinarlo con IEX/Massive como si fuera el mismo feed. |
| Replay histórico de 400 sesiones | **Massive Stocks Basic**, confirmado por `output/historical_replay/input/massive_400_2026-09-25/manifest.json` | Investigación histórica | El manifiesto no dice Alpaca SIP. Un reporte anterior que lo llamaba SIP era incorrecto. |
| Backfill degradado | Alpaca IEX histórico consultado después de los cortes | Diagnóstico de los cortes perdidos | No es evidencia PIT ni prospectiva; siempre excluido de entrenamiento y cobertura válida. |

La cuenta de Alpaca devolvió HTTP 403 para SIP contemporáneo y HTTP 200 para IEX en el preflight del 30-09; una nueva consulta de lectura SIP 5m con `end` a cinco minutos del presente devolvió **HTTP 403** el 05-10. [Alpaca documenta](https://docs.alpaca.markets/us/docs/market-data-faq) que el plan gratuito ofrece IEX en tiempo real y SIP histórico solo si el `end` es al menos 15 minutos anterior. Por tanto, el SIP histórico **no sustituye** la vela 5m de un corte prospectivo a las 10:00 sin retrasar el corte. No se ha cambiado silenciosamente el colector. Cualquier migración a SIP requerirá acceso real a ese corte, nueva versión de política, cohorte independiente y validación de equivalencia.

## Mapa de scores · `CONTINUOUS_CONTEXT_LOGIT_V2`

Antes, cada EMA/MACD votaba solo `+peso` o `-peso`. Cinco votos positivos daban exactamente sesgo `+1` aunque las distancias al precio fueran distintas. La última conversión imponía además un suelo/techo de 4 puntos, produciendo repetidamente `70/26/4` (71 de 144 filas observadas en la auditoría). Era una pérdida de resolución **matemática**, no evidencia de una probabilidad del 70%.

Ahora el mapa de **visualización** usa, para cada EMA, `peso × (precio − EMA)/(abs(precio − EMA) + ATR)` y para MACD la misma forma con escala `0.25 × ATR`; promedia por pesos. Esto preserva dirección y magnitud relativa sin que una alineación fuerte se vuelva automáticamente `±1`. El bloque direccional se reparte mediante `sigmoid(3 × sesgo)`; no hay recorte final a un suelo fijo. La categoría rango mantiene su fórmula y el total sigue sumando 100 después de redondeo. Se registra `score_map_version` en nuevas capturas; las antiguas permanecen firmadas con su versión implícita V1.

**No es calibración empírica.** El mapa muestra scores `PRELIMINAR` `/100`, separados de la frecuencia de tocar cada nivel desde el corte hasta el cierre. Cambiar su resolución no valida entradas, no modifica el gatillo, las barreras, los umbrales de decisión ni el estado de aprobación. Es incorrecto interpretar una fila alcista y una probabilidad de tocar soporte como sucesos complementarios.

Auditoría reproducible, sin entrenamiento: `python scripts/audit_score_map_resolution.py`. Tras el corte del 05-10 a las 12:30 NY, sobre 26 análisis archivados IEX × seis horizontes (156 filas), la versión V2 produjo **98 ternas distintas y 0 filas exactamente 70/26/4**, sin errores de reproducción. Esto demuestra mayor resolución descriptiva, **no** mayor capacidad predictiva ni calibración.
