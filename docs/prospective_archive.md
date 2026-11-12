# Archivo PIT prospectivo — uso y límites

Los scripts nuevos no importan Streamlit, no escriben en `portfolio.db`, no ejecutan órdenes y no entrenan modelos. El archivo vive en `output/prospective_archive/`, ignorado por Git. Guarda los titulares **tal como el colector los vio** junto con su hora real de recepción, el modelo FinBERT congelado, features técnicas existentes, velas IEX ajustadas por splits y contratos de barrera. Cada predicción queda inmutable y firmada con SHA-256; el resultado se añade en un **sidecar firmado** que referencia el hash del snapshot. Esto equivale a actualizar el registro lógico sin alterar lo que se sabía en el corte.

## Cortes y causalidad

- XNYS: 10:00, 11:00, 12:30, 14:00 y 15:30, zona `America/New_York`. Solo se acepta una captura desde el instante del corte hasta menos de cinco minutos después. No se recupera ni se fabrica un corte perdido.
- El campo `scheduled_cut_at_utc` identifica el horario predefinido; `cut_at_utc` es el **instante real** en que terminaron adquisición, análisis y score. Cada titular lleva `published_at_utc` y `collected_at_utc`; el segundo es nuestra evidencia de disponibilidad. El primero solo define las ventanas de 4h/24h. Ausencia de titulares se guarda como `N/D`, nunca como sentimiento neutro.
- Se exige la vela cerrada exactamente al corte, no una vela abierta o anterior. Mercado: Alpaca **IEX** 5m y diario, ambos `adjustment=split`, en la versión `XNYS_FIVE_CUT_ALPACA_IEX_RESEARCH_V1`. La cuenta devolvió SIP HTTP 403 e IEX HTTP 200 en el preflight del 30-09-2026. IEX cubre una sola bolsa: **sus volúmenes, OHLC y features no son directamente comparables con el replay histórico de Massive Stocks Basic** (manifiesto firmado `output/historical_replay/input/massive_400_2026-09-25/manifest.json`). No era Alpaca SIP; la documentación anterior lo llamaba así por error. Si más adelante hay acceso SIP en tiempo real, debe abrirse una cohorte/version nueva, sin mezclar datos. [Alpaca explica la diferencia de feeds](https://docs.alpaca.markets/us/docs/market-data-faq). Noticias: [Finnhub Company News](https://finnhub.io/docs/api/quote), cuyo `datetime` es hora de publicación, no disponibilidad histórica; la disponibilidad prospectiva la aporta nuestra recepción.
- El contrato operativo vigente solo autoriza el anclaje 11:00. Los otros cuatro cortes son **investigación no accionable**; ningún script cambia ese contrato. Las cinco observaciones de un día, y los horizontes 1h/1D solapados, no deben contarse como cinco operaciones independientes.

## Windows: preparación y programación

Usa el mismo usuario de Windows que abrirá sesión durante el horario de la PC. El entorno de investigación existente tiene FinBERT/Transformers. En PowerShell, dentro de STOCK:

```powershell
.\output\historical_replay\finbert_research_env\Scripts\python.exe .\scripts\prospective_credentials.py setup
```

El programa pedirá **ocultas** las claves vigentes `FINNHUB_API_KEY`, `APCA_API_KEY_ID` y `APCA_API_SECRET_KEY`. No las escribas en comandos ni en el chat. Se guardan únicamente como bytes cifrados con DPAPI para el usuario actual, en una ruta ignorada por Git. Para rotarlas: repite con `setup --replace`. No uses claves ya expuestas para una automatización duradera.

Después ejecuta:

```powershell
.\scripts\register_prospective_tasks.ps1
```

Quedan cuatro tareas: colector vigilante cada minuto, etiquetador cada 15 minutos, cobertura cada hora y `STOCK-Prospective-Cut-Wake`, que despierta Windows de suspensión tres minutos antes de cada posible corte local. El guard XNYS sigue eligiendo las cinco horas reales; en cambios de horario de verano puede haber marcadores de despertar extra, pero no snapshots extra. `wake.log` conserva cada disparo. Usan `pythonw.exe` sin ventana; los errores de arranque quedan en `output/prospective_archive/logs/scheduler.log`. El colector **solo genera datos en los cinco cortes XNYS**; fuera de las ventanas sale sin consultar APIs. Task Scheduler no abre la app. Las tareas están limitadas al usuario interactivo para que DPAPI pueda descifrar. **WakeToRun despierta de suspensión, no enciende una PC apagada**; el usuario debe mantener disponible la sesión y permitir temporizadores de activación en Windows. El etiquetador sí puede ponerse al día tras el arranque porque usa resultados posteriores, no predicciones retrospectivas.

El 01-10-2026 y 05-10-2026 el equipo estaba en suspensión a las 10:00 NY y despertó 46 y 51 minutos después. Los dos cortes faltantes se reconstruyeron **solo para diagnóstico** en `output/prospective_archive/backfills/`, con `backfilled=true`, `pit_quality=degraded`, `training_eligible=false` y `production_actionable=false`. `cut_at_utc` conserva la hora real de reconstrucción, mientras `historical_cut_at_utc` marca el corte perdido. No se reconstruyó la disponibilidad de noticias ni fills. Los ocho registros firmados (dos fechas × dos activos × dos horizontes) siguen **excluidos** del denominador de cobertura prospectiva, del etiquetador y del entrenamiento. Nunca se reescriben snapshots reales.

Verificación manual en una ventana válida:

```powershell
.\output\historical_replay\finbert_research_env\Scripts\python.exe .\scripts\prospective_credentials.py run collector
.\.venv\Scripts\python.exe .\scripts\prospective_coverage.py
```

Revisa `output/prospective_archive/logs/collector.log` y `labeler.log`. Para un ejemplo de esquema **sintético, no contado**: `.\.venv\Scripts\python.exe .\scripts\collect_prospective_snapshots.py --example`.

## Linux y macOS

Inyecta las tres variables desde tu gestor de secretos al proceso (nunca pongas valores en `crontab`). Instala las dependencias de investigación y verifica el snapshot local de FinBERT. Cron puede ejecutar el colector cada minuto y el etiquetador cada 15; el guard interno XNYS resuelve festivos y horario de verano:

```cron
* * * * * /ruta/python /ruta/STOCK/scripts/collect_prospective_snapshots.py
*/15 * * * * /ruta/python /ruta/STOCK/scripts/label_prospective_snapshots.py
0 * * * * /ruta/python /ruta/STOCK/scripts/prospective_coverage.py
```

Configura el programador para impedir solapamientos de una misma tarea. No uses horarios locales fijos como reemplazo del guard XNYS.

## Etiquetas y cobertura

El etiquetador espera el vencimiento XNYS más 15 minutos, recupera velas posteriores IEX y usa el etiquetador de triple barrera existente. Ante evidencia incompleta permanece `PENDING`; el último precio actual nunca sustituye una vela faltante. `mfe`/`mae` son excursiones OHLC desde la **entrada de referencia**, no P&L de un fill real. `fill_possible` significa solo alcance conservador por OHLC; se muestra separado de la autorización de producción y del fill confirmado por un corredor.

El reporte separa snapshots, noticias, etiquetas, planes válidos, entradas autorizadas, fills posibles y días agrupados. La fecha estimada para 300 permanece `N/D` hasta contar al menos cinco sesiones observadas; aun después es una proyección, no promesa. Llegar a 300 snapshots no equivale a 300 eventos independientes ni a 300 entradas ejecutables.

## Evaluación forward del piloto LONG 1D congelado

`scripts/train_prospective_pilot.py forward` evalúa los modelos `RIDGE_TECH` y
`RIDGE_TECH_NEWS` del último run V2 firmado **sin reentrenarlos**, sobre las
sesiones posteriores al preregistro (`session_ny` mayor que la última sesión
del protocolo y corte posterior al congelamiento). Reutiliza las medianas,
medias y escalas del entrenamiento; las etiquetas que llegaron después del
congelamiento no cuentan como entrenamiento. El resultado se firma en
`output/prospective_research/long_1d_pilot_v2/forward_<UTC>.json`.

Es la primera evidencia fuera de muestra real del piloto, pero con pocas
sesiones sigue siendo informativa: no autoriza compras ni cambia el gatillo.

```powershell
.venv\Scripts\python.exe scripts\train_prospective_pilot.py forward
```
