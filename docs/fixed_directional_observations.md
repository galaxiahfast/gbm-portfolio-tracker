# Observaciones automáticas y objetivo operativo (corte NY 11:00)

El trabajo Windows `GBM_Forward_Collector` ejecuta `scripts/daily_auto_collector.py
--scheduled --symbols SMCI NVDA` cada sesión XNYS a las 11:00 de Nueva York.
El disparador tiene dos candidatos UTC para el cambio de horario; el proceso
solo acepta la ventana **11:00–11:20 NY** de una sesión real. Si la PC está
apagada o no hay una vela reciente cerrada, no se inventa una observación ni
se retrofecha. `observed_at` es la hora real de emisión.

Por activo y sesión se guardan, en una sola transacción, seis contratos
firmados SHA-256 en `live_model_observations`: 1h, 6h, 1D, 1S, 1M y 6M.
Todos usan el mismo análisis headless que las zonas. El protocolo actual es
`XNYS_1100_OPERATIONAL_TARGET_V2`; `XNYS_1100_WINDOW_V1` queda admitido solo
para leer y auditar emisiones históricas. Las antiguas emisiones dependientes
de abrir Streamlit no se mezclan con la cohorte actual. Un reintento diario
íntegro devuelve cero filas nuevas. Un corte parcial o alterado bloquea la
escritura.

Cada fila congela dos contratos distintos y no intercambiables:

1. `scenario_contract`: clasificación diagnóstica del **cierre al vencimiento**
   en `UP / RANGE / DOWN`. Se conserva por compatibilidad y por su serie de
   calibración histórica.
2. `operational_contract`: objetivo primario de validación
   `TP_FIRST / SL_FIRST / TIMEOUT`. Congela dirección, precio de referencia,
   TP1, stop, vencimiento, políticas y revisión del etiquetador. Los seis
   horizontes comparten el plan observado; cada uno tiene su propio timeout
   XNYS.

El vector direccional `UP / RANGE / DOWN` se guarda únicamente como procedencia.
No se transforma ni se presenta como probabilidad de tocar primero TP o SL:
esos eventos contienen información de trayectoria que un cierre final no
identifica.

`GBM_Forward_Resolver` y `GBM_Forward_Catchup` mantienen dos resoluciones
separadas. El contrato direccional usa su cierre XNYS exacto. El contrato
operativo recorre causalmente la trayectoria cerrada y puede resolverse antes
del timeout si una barrera es alcanzada. Si falta evidencia completa, permanece
pendiente; nunca se usa el precio actual como sustituto. Ambas resoluciones son
independientes del log de seis zonas.

## Contrato causal TP primero / SL primero / timeout

La trayectoria operativa empieza en `observed_at.ceil("5min")`: la apertura de
la primera vela de 5 minutos que queda **completamente después** de la emisión.
La vela que ya estaba en formación al emitir se excluye, aunque su cierre se
conozca después. Solo se consideran velas regulares XNYS cerradas y con OHLCV
válido.

- `TP_FIRST`: el take profit congelado fue la primera barrera alcanzada.
- `SL_FIRST`: el stop congelado fue la primera barrera alcanzada.
- `TIMEOUT`: ninguna barrera fue alcanzada; la salida se valora al cierre exacto
  del vencimiento.

Las reglas conservadoras están versionadas en el contrato:

- Gap adverso a través del stop: salida al precio real de apertura del gap.
- Gap favorable a través del TP: salida al TP congelado, sin atribuir una mejora
  de ejecución no demostrada.
- TP y SL dentro de la misma vela sin orden intrabar observable: `SL_FIRST`.
- El criterio anterior también se aplica al fallback diario, donde el orden
  intradía de máximo y mínimo no puede reconstruirse.

La fuente primaria es una secuencia 5m completa por sesión. Si ya terminó una
sesión futura y falta su intradía, puede utilizarse su OHLC diario como fallback
conservador. El fallback diario **nunca** reconstruye la sesión en que se emitió
el pronóstico, porque su máximo y mínimo incluyen horas anteriores a la
observación. Una sesión parcial, una vela faltante o un cierre de timeout no
observable deja el resultado pendiente en vez de imputarlo.

El resultado operativo se guarda en `operational_model_outcomes`, enlazado por
`observation_id`. El contrato es inmutable; una transición de `PENDING` a
`RESOLVED` exige outcome, precio, hora, fuente, evidencia y firma completos.
La firma de resolución incorpora el hash de la observación padre, el hash del
contrato, el resultado, el SHA-256 de las velas inspeccionadas y las referencias
canónicas a los artefactos de mercado. Los triggers impiden reescribir o borrar
una resolución terminada.

La migración analítica 12 añade un checkpoint causal firmado a cada resultado
pendiente. Después de verificar un tramo continuo, conserva el último cierre
procesado, el número acumulado de velas, un hash encadenado por barra y las
referencias de sus artefactos. La siguiente ejecución empieza exactamente en
ese cursor y solo necesita las velas nuevas. Así los horizontes 1M/6M no
dependen de que la caché de 5 minutos conserve seis meses completos. El cálculo
de calendarios y barras se realiza fuera de la transacción SQLite; el checkpoint
o resultado se confirma con un `UPDATE` compare-and-swap breve.

`TP_FIRST / SL_FIRST / TIMEOUT` es por ahora una **etiqueta observada**, no una
probabilidad operativa calibrada. El contrato declara
`operational_probabilities = null` y
`N/D_HASTA_CALIBRACION_EMPIRICA_FIRST_PASSAGE`. Solo después de reunir muestra
suficiente, separar cronológicamente calibración y holdout, y medir Brier OOS
por activo, horizonte y versión, podrán publicarse probabilidades operativas.

Comprobación sin descargar ni escribir en la base:

```powershell
.\.venv\Scripts\python.exe scripts\daily_auto_collector.py --check-only --scheduled --symbols SMCI NVDA
```

Si la tarea no está instalada, `scripts/install_autopilot_tasks.ps1
-CollectorOnly -Preview` permite revisar su definición sin registrarla;
`-CollectorOnly` instala solo el colector. Los trabajos no ejecutan órdenes ni
modifican efectivo, operaciones o posiciones reales.

## Registro reproducible de cada corte

Desde el siguiente corte real, las seis observaciones comparten un `run_id`
determinista por símbolo, sesión y protocolo. Cada fila SHA-256 contiene el
mismo snapshot `replay` de features observadas (incluido el contexto cruzado),
las últimas lecturas de indicadores por temporalidad, huellas SHA-256 del
código del motor y referencias verificables a los archivos históricos 5m/1d.
También congela el vector de cierre, los objetivos y bandas de cada horizonte,
y el contrato operativo first-passage. El cierre direccional se firma en la
fila padre; TP/SL/timeout se firma en su fila hija inmutable. Un resultado puede
resolverse mientras otros horizontes siguen pendientes. El identificador común
no altera los vencimientos XNYS ni crea resultados que todavía no se conocen.

Para inspeccionar un corte sin modificar la base:

```python
from portfolio_tracker.db import Database
from portfolio_tracker.config import DATA_DIR
from portfolio_tracker.repository import PortfolioRepository
from portfolio_tracker.services.model_execution_record import execution_id
from portfolio_tracker.services.directional_collection import COLLECTION_PROTOCOL
from scripts.autopilot_market_cache import MarketCache

repo = PortfolioRepository(Database())
run_id = execution_id("SMCI", "2026-09-21", COLLECTION_PROTOCOL)
record = repo.live_model_execution_record(run_id)
assert record is not None  # None significa lote ausente, incompleto o alterado
verified_inputs = MarketCache(DATA_DIR / "autopilot" / "market").verify_artifact_refs(
    "SMCI", record["input_artifacts"]
)
```

La ruta de caché debe ser la utilizada realmente por el programador de tareas.
Si no existen ambos archivos, el corte se etiqueta
`FEATURE_SNAPSHOT_ONLY`: sigue siendo auditable a nivel de features, pero **no**
se presenta como replay bit a bit de los datos crudos. La serie cruda de NVDA
usada como peer tampoco se archiva aún en este contrato; se congela su contexto
derivado y se declara explícitamente esa limitación. Los cortes anteriores no
se rellenan retroactivamente ni reciben hashes inventados.

SHA-256 detecta cambios accidentales o alteraciones sin recomputar la firma;
no autentica por sí solo frente a alguien con acceso para reescribir la base y
todas sus firmas. Tampoco convierte los scores heurísticos en probabilidades
empíricamente calibradas. En particular, la nueva etiqueta first-passage debe
acumular su propia cohorte OOS; no puede tomar prestada la calibración de cierre.
