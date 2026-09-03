# F09 — Compatibilidad de fundamentales
Fecha: 2026-09-02

## Alcance
Cambio exclusivamente analítico en `analytics/fundamental_news.py`.
No requiere migraciones, modificaciones de repositorio ni acceso a la base contable.
Los snapshots nuevos usan `fundamental-news-v3-period-aligned`.
Los snapshots anteriores siguen siendo legibles y no se reescriben ni se vuelven
a firmar. Para obtener los ratios corregidos se necesita un nuevo corte fundamental;
un PDF o snapshot antiguo conserva sus valores originales.

## Correcciones
- Conversión a caja = OCF / beneficio neto; margen FCF = FCF / ingresos.
  Solo se calculan con QUARTER/QUARTER o TTM/TTM, misma fecha de cierre,
  mismo inicio cuando se informa y misma moneda.
- Las unidades monetarias y su escala deben estar declaradas. Escalas positivas
  diferentes se normalizan antes de dividir; monedas diferentes no se convierten
  implícitamente.
- La descarga usa estados trimestrales explícitos, valores monetarios raw y
  financialCurrency. No usa la moneda de cotización como sustituto.
- Los totales del resumen info sin metadatos no se dividen entre cifras trimestrales.
  El ratio usa exclusivamente el par trimestral verificado, o queda N/D.
- Se selecciona la columna de fecha más reciente, independientemente de su orden.
  Una celda ausente no permite reutilizar silenciosamente un trimestre anterior.
  Fechas o filas ambiguas tampoco producen ratios.
- Cero es un dato, no ausencia. Un numerador cero produce ratio cero;
  denominadores cero o negativos producen N/D en conversión a caja y margen FCF.
  OCF negativo sobre beneficios positivos conserva su penalización.
- FCF cero tiene impacto neutral. Deuda/capital negativo no recibe el bonus
  de bajo apalancamiento. La cobertura de intereses verifica también la pareja
  temporal; admite el signo contable del gasto sin convertir EBIT negativo
  en una bonificación.

## Contrato para otras fuentes
Los DataFrames de estados deben declarar en `frame.attrs`:
`period` (QUARTER o TTM), `currency` (moneda financiera),
`unit="currency"` y `scale` (por ejemplo 1 o 1000).
La fecha de la columna es el fin del periodo. `period_start` es opcional,
pero si se proporciona debe coincidir en ambos componentes.

El argumento opcional `financial_metadata` permite datos info verificados.
Ejemplo de metadatos para cada clave:
`{"period": "TTM", "period_end": "2026-06-30", "currency": "USD",
"unit": "currency", "scale": 1}`.
Las parejas son `operatingCashflow/netIncomeToCommon` y
`freeCashflow/totalRevenue`. Si se declara metadata para una pareja,
ambos componentes deben pasar validación: no se busca otro periodo para
rescatar una puntuación favorable.

## Trazabilidad
En metrics se guardan `cash_conversion_basis`, `fcf_margin_basis`,
sus numeradores y denominadores normalizados, y `interest_coverage_basis`.
Las razones del snapshot incluyen la base o el motivo de exclusión.
Estos campos entran en el payload JSON canónico existente y, por tanto,
en su integridad SHA-256 cuando se almacena por la vía normal.

Los campos de resumen free_cash_flow/operating_cash_flow conservan su prioridad
de fuente (info si existe, estado si falta); no implican que ese dato sea el
numerador del ratio. El numerador real queda identificado explícitamente.
Las métricas que el proveedor entrega ya calculadas (márgenes y crecimiento)
no se recalculan en esta corrección; validar sus componentes originales
requeriría datos adicionales.

## Verificación
Regresiones nuevas en `tests/test_fundamental_periods_f09.py`, sin modificar
las pruebas existentes. Las pruebas de descarga usan un proveedor simulado y
directorio temporal; no consultan cuentas ni generan órdenes.

Referencia del adaptador: [API oficial yfinance: get_income_stmt](https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.get_income_stmt.html)
distingue frecuencias yearly, quarterly y trailing.
[Implementación oficial de estados](https://github.com/ranaroussi/yfinance/blob/main/yfinance/scrapers/fundamentals.py)
utiliza reportedValue.raw.

