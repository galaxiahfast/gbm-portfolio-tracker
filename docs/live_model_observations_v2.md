# Observaciones en vivo v2 - F01/F02

> Nota de vigencia: este documento conserva el contrato histórico V2 para
> auditoría. Las emisiones automáticas actuales usan
> `XNYS_TRADING_MINUTES_AND_FUTURE_SESSIONS_V3` y el protocolo
> `XNYS_1100_OPERATIONAL_TARGET_V2`. V2 sigue siendo verificable, pero no se
> mezcla con la cohorte vigente.

## Alcance y migración

La migración 9 agrega únicamente columnas, índices y restricciones a
`live_model_observations`. No reconstruye tablas, no modifica operaciones,
efectivo, comprobantes, órdenes ni estado de posiciones. Usa el mecanismo
existente de copia previa a migración. Se aplica idempotentemente al iniciar.

Las filas anteriores conservan sus datos y hashes originales. Quedan como
`LEGACY_UNVERIFIED`: no se re-firman resultados históricos cuya procedencia
no puede demostrarse. Se excluyen de estadísticas y calibración y aparecen
como no verificables en auditoría. El contador de muestras válidas puede bajar.

## Contrato temporal explícito

- `analysis.as_of`: etiqueta de **apertura** de la vela 5m, se conserva para no
  alterar los consumidores de estado operativo.
- `analysis.source_bar_closed_at` / `source_bar_at` persistido: **cierre** de
  esa vela, instante en que el precio de referencia ya era conocible.
- `observed_at`: momento real de emisión del pronóstico, UTC explícito.
- `available_at`: vencimiento inmutable, en minutos de reloj desde la emisión,
  redondeado **hacia arriba** al siguiente cierre 5m. Ejemplo: emisión 15:00:17
  con horizonte 60 minutos vence a las 16:05:00. No se retrofecha la emisión.
- `outcome_bar_at`: cierre histórico empleado; debe ser exactamente
  `available_at`.
- `resolved_at`: momento de procesamiento; puede ser posterior al vencimiento
  sin cambiar el precio que se utiliza.

Política versionada: `WALL_CLOCK_CEIL_5M_XNYS_V2`. Una hora son 60 minutos;
6 horas son 360; día/semana son 1.440/10.080 minutos; mes y 6 meses se aproximan
a 30 y 180 días de calendario. **No son horizontes de minutos de trading.**
No se desplaza un vencimiento a una sesión posterior: se invalida si cae
fuera del horario regular XNYS, en fin de semana, festivo o después de un
cierre anticipado. El cambio EST/EDT lo resuelve el calendario bursátil.

## Resolución causal

El repositorio recibe OHLCV bruto 5m con timestamps de apertura y zona horaria,
no indicadores filtrados ni cotización spot. Solo acepta la vela cuyo cierre
coincide exactamente con el vencimiento y ya ocurrió en `current_as_of`.
No hay búsqueda por cercanía, interpolación, forward-fill ni sustitución
por la apertura del lunes. El argumento antiguo `current_price` lanza error.

- Mercado cerrado al vencimiento: `INVALID_MARKET_CLOSED`, sin outcome y con
  firma de invalidación; no participa en calibración.
- Falta la vela exacta o es inválida/duplicada: permanece `PENDING`; puede
  resolverse posteriormente si se suministra ese histórico exacto.
- La descarga intradía actual tiene retención limitada. Un horizonte largo
  sin histórico disponible no se resuelve; no se inventa una muestra.
- Se rechaza emitir sobre un precio con 5 minutos o más de antigüedad desde
  el cierre de su vela. Se deduplican reejecuciones por símbolo, vela fuente
  y horizonte para no inflar artificialmente el tamaño muestral.

## Integridad e inmutabilidad

`observation_sha256` cubre el pronóstico y su contrato temporal/versión.
`resolution_sha256` cubre todos esos campos, el hash original, los cuatro
campos de resultado solicitados, el cierre utilizado, la fuente y el estado.
Ambos se verifican antes de usar una muestra. El hash inicial nunca cambia.

La resolución es atómica bajo `BEGIN IMMEDIATE` y solo ocurre desde PENDING.
Triggers SQLite bloquean modificar el pronóstico, borrar una fila v2 o
reescribir una resolución finalizada. Una alteración fuera de esos controles
también se detecta al verificar hashes y excluye automáticamente la muestra.
Una fila pendiente alterada no se puede convertir en una resolución firmada.

SHA-256 **no es autenticación** contra quien pueda reescribir simultáneamente
datos, triggers y hashes; no sustituye controles de acceso ni respaldos.

## Verificación

`tests/test_live_model_integrity.py` cubre alteraciones, reescrituras,
idempotencia, viernes/lunes, festivos, cierres anticipados, DST, duplicados,
velas futuras, datos ausentes y preservación del legado/contabilidad.
Se actualizó la antigua prueba que aceptaba un precio spot tardío para que
exija la vela histórica del vencimiento. El generador de auditoría conserva
su catálogo histórico y marca fuentes cambiadas para revalidación humana.

## Objetivo operativo posterior, separado de V2

Las emisiones actuales conservan la resolución direccional de cierre en la
fila padre y añaden una fila hija inmutable en `operational_model_outcomes`.
Su etiqueta primaria es `TP_FIRST / SL_FIRST / TIMEOUT`; no reemplaza ni
reinterpreta resultados V2 existentes.

La evaluación first-passage comienza en la primera vela 5m completamente
posterior a `observed_at`, usa gaps adversos al precio de apertura, limita gaps
favorables al TP y resuelve como `SL_FIRST` cualquier empate intrabar. La fuente
principal es 5m; una sesión futura completa puede usar OHLC diario bajo esa
misma política conservadora, pero la sesión de emisión nunca se reconstruye con
su vela diaria. Resultado, evidencia inspeccionada y referencias de artefactos
quedan firmados.

No existe una conversión válida entre la probabilidad `UP / RANGE / DOWN` y la
probabilidad de cuál barrera se toca primero. Por ello las probabilidades del
nuevo objetivo se mantienen `N/D` hasta reunir y evaluar una cohorte OOS propia.
