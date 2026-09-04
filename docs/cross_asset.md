# Correlación SMCI / NVDA

La vista ejecutiva y la técnica conservan su navegación. Ambas añaden **Correlación · Relación con NVDA** al analizar SMCI y la relación inversa al analizar NVDA. Los cuatro PDF usan el mismo contexto de la pantalla: ejecutivo, técnico, combinado y maestro.

## Qué se calcula

- Pearson sobre **20 retornos diarios** de cierres, no correlación de niveles de precio. Exige 21 cierres completos, alineados por sesión NYSE; no rellena huecos.
- Ratio **SMCI/NVDA** al último cierre diario completo, media de 50 sesiones y desviación porcentual. Mantiene este numerador incluso en la vista NVDA.
- RSI14 Wilder diario de cada activo. No interpreta automáticamente sobrecompra como señal bajista ni confunde RSI con fuerza relativa de retornos.
- Nuevos máximos/mínimos de las últimas 20 velas previas, por mecha, en diario y en 5 minutos. Señala divergencias sin declararlas una reversión cierta.
- Adelantos intradía: ruptura **por cierre**, volumen de la vela mayor a **1.2 veces** el promedio de las 20 anteriores. Compara además la dirección del retorno de los últimos 30 minutos.

Sólo se usan velas cerradas. La comparación intradía exige las mismas 21 velas completas en ambos activos y el mismo instante de cierre. El promedio de volumen y los extremos pueden incluir el final de la sesión anterior (permite analizar a las 11:00, cuando sólo hay 18 velas del día). El momentum exige 30 minutos íntegros de la sesión actual y nunca confunde el salto nocturno con 30 minutos. Antes de reunir esos datos, en festivos, con datos atrasados o fuera de sesión no aplica ajuste intradía. Puede seguir mostrando la correlación diaria con su fecha explícita.

## Impacto prudente

Con correlación diaria >0.70:

| Evidencia del activo par | Propuesta de ajuste al score alcista |
|---|---:|
| Rompe resistencia con volumen y el activo analizado aún no confirma | +5 puntos |
| Pierde soporte con volumen y el activo analizado aún no confirma | -5 puntos |
| Retornos de 30 minutos en sentidos contrarios | +3 / -3 según el par |
| Sin adelanto/divergencia, correlación insuficiente o no válida | 0 |

El impacto aplicado puede ser menor: no invierte un sesgo existente, respeta los límites 15–85, no bonifica LONG bloqueado y no se aplica sobre veto de riesgo o estado activo recibido. Se muestra **propuesto / aplicado** para auditar esas diferencias. Se reemplaza el componente anterior al repetir el enriquecimiento: nunca acumula bonos por recarga.

Este ajuste afecta **solamente al score heurístico contextual**. No modifica probabilidades de toque/cierre de las seis zonas, horizontes calibrados, stops, objetivos, gatillos, permisos de operación, exposición ni gestión de posiciones. No es una probabilidad calibrada ni una nueva señal de compra. Los pesos son provisionales hasta contar con evidencia prospectiva suficiente.

## Descarga y fallos

`services/cross_asset.py` descarga el par mediante yfinance, sin importar Streamlit. Usa un año diario y cinco días a 5m, con caché máxima de cinco minutos, renovación por bucket cerrado, deduplicación por activo y hasta dos trabajadores daemon. La UI inicia la consulta mientras calcula el activo principal. La espera adicional está limitada a dos segundos; una consulta pendiente se recoge en una actualización posterior. Los fallos se cachean 60 segundos para evitar ráfagas.

Si no hay datos utilizables, se muestra **Correlación no disponible**, sin alterar el score primario. Datos diarios válidos con intradía incompleto se muestran sin bono. Yahoo puede presentar retrasos: se muestran cortes cerrados, no ticks en tiempo real garantizados.

## Integridad y validación

El contexto completo (correlación, fechas, ratio, RSI, divergencias, huella de entradas y ajuste) se registra en `zone_prediction_log.context_json.cross_asset`. Este campo ya está cubierto por `forecast_sha256`; no se requiere migración ni columna nueva. Los registros anteriores no se rellenan retrospectivamente. La nueva versión del modelo cambia su hash y conserva separadas las cohortes previas.

El colector headless y la UI llaman a `enrich_cross_asset` antes de congelar el snapshot de seis zonas. Ninguna función nueva escribe órdenes, efectivo, posiciones contables o credenciales.

VALIDACIÓN añade **Aciertos globales · todas las emisoras** sin quitar los filtros por símbolo. Desglosa versión y evento Toque/Cierre, muestra recuentos y Brier con igual peso por sesión. “Ocurrieron” cuenta eventos alcanzados, no operaciones rentables; seis zonas y dos acciones correlacionadas no son ocho muestras independientes.

## Uso de las funciones puras

```python
from portfolio_tracker.analytics.cross_correlation import calculate_rolling_correlation

# DataFrames OHLCV diarios previamente cerrados; no hay descargas ocultas aquí.
histories = {"SMCI": smci_daily, "NVDA": nvda_daily}
correlation = calculate_rolling_correlation("SMCI", "NVDA", period=20, histories=histories)
```

`calculate_price_ratio(histories=...)`, `get_relative_strength(histories=...)` y `detect_divergence(histories=...)` admiten los mismos datos explícitos. Para la divergencia, ambos DataFrames deben tener índices idénticos y al menos 21 velas.

## Comprobación local

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cross_correlation.py -q
.\.venv\Scripts\python.exe scripts/check_cross_asset.py --symbol SMCI --pdf output/pdf/correlacion_SMCI.pdf
```

La segunda instrucción consulta datos reales y genera un PDF **sólo de contexto cross-asset** más un JSON con evidencia de entrada. No abre ni modifica SQLite, no registra predicciones retrospectivas y no constituye una prueba de acierto estadístico. Para ver los cuatro informes integrados, actualizar el análisis en la app; los PDF descargados anteriormente no se reescriben.
