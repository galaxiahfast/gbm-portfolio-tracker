# F07/F08 — Riesgo marcado a mercado y vigencia técnica

Fecha: 2026-09-02.

## F07

Un stop LONG saltado por una apertura inferior se ejecuta a esa apertura;
un stop SHORT saltado por una apertura superior también. Los gaps conocidos
se procesan antes que los extremos intravela. Las comisiones y el slippage
configurados se cargan una sola vez como costes explícitos, sin ocultarlos
en un fill idealizado ni duplicarlos en el precio.

Cada operación conserva `mark_to_market` (timestamp de vela, fase y P&L neto
de liquidación) y `maximum_adverse_excursion_usd`. Se valoran apertura,
extremos y cierre mientras está abierta. El último mark liquida la posición;
no se utilizan máximos/mínimos posteriores a una salida por gap. Los marks
simultáneos de distintos activos se agregan antes de medir el patrimonio.
`maximum_drawdown_pct` y la curva OOS ya incluyen pérdidas flotantes, aunque
la operación termine en ganancias. El JSON auditado incluye esta trayectoria.

**Límite de OHLC:** sin ticks no se conoce el orden real de máximos y mínimos.
En velas sin salida se usa favorable→adverso como envolvente conservadora del
drawdown. En velas que tocan stop y TP se conserva stop primero. No se afirma
reconstrucción exacta tick a tick ni se cuentan precios posteriores al fill.
Los payloads anteriores sin marks siguen siendo legibles con su información
de cierre, pero no se inventan excursiones intradía para esos registros.

## F08

`TechnicalDataVeto`, compatible con ValueError, informa estado `UNKNOWN`,
`risk_veto=True`, `activation_trigger_met=False` y motivo explícito. No se
construye un pronóstico sustituyendo NaN por porcentajes aparentes. Streamlit
muestra `UNKNOWN / INDEFINIDO` y detiene ese análisis antes de registrar nuevos
pronósticos o presentar una entrada.

- Precio constante durante 14 intervalos: veto, incluso si una EMA conserva
  memoria de movimientos antiguos. Un RSI 0/0 es indefinido, no RSI=100.
- Un RSI=100 con ganancias reales y sin pérdidas sigue siendo matemáticamente
  válido; un StochRSI sin rango de RSI sigue siendo indefinido y veta el análisis.
- Se comprueban datos OHLCV y los indicadores de la última vela y su precedente.
  No se permite que dropna() retroceda silenciosamente a una vela anterior.
  Se usa únicamente el sufijo válido continuo, sin juntar extremos separados
  por huecos de indicadores.
- ADX diario se calcula con todo el historial cerrado antes del recorte de
  calentamiento de Ichimoku; no se genera un NaN artificial por truncarlo antes.
- La interfaz en vivo llama con `require_fresh=True` y reloj explícito. Se exige
  el último cierre 5m y la última sesión diaria cerrada según XNYS, incluyendo
  festivos, horario de verano y sesiones cortas. Antes del primer cierre 5m
  de la sesión se espera. Llamadas de investigación offline pueden omitir
  vigencia respecto al reloj, pero nunca validez ni integridad del último corte.

No se modifican tablas contables, órdenes, cifrado ni registros reales. No hay
migración. Se agregan pruebas aisladas; ninguna prueba funcional anterior se
reescribe. Reiniciar Streamlit carga la validación actualizada.
