# Autopiloto de evidencia · Windows / Nueva York

Implementado el 3 de septiembre de 2026. No ejecuta operaciones de trading.

## Estado operativo y límite estadístico

El autopiloto está **OPERATIVO EN MODO RECOLECCIÓN**. La resolución conserva las velas
de 5 minutos completas para evaluar toques y utiliza el cierre diario oficial para el
resultado de cierre. Si ambas fuentes difieren más de USD 0.01, no se descarta la sesión:
se guardan las dos cifras, la diferencia y una advertencia firmada en la resolución.

Los resultados siguen siendo preliminares. **Se requieren al menos 30 sesiones
independientes por activo y versión para declarar calibración OOS concluyente.** Una fila
por zona no equivale a una sesión independiente, y una zona alcanzada antes del corte se
excluye del Brier de toque en lugar de imputarse artificialmente como acierto o fallo.

## Activación (una sola vez)

1. Verifica que esta copia del proyecto sea la que seguirás usando. El instalador guarda rutas absolutas y utiliza su propio entorno Python.
2. Haz clic derecho en **scripts\instalar_autopiloto.bat → Ejecutar como administrador**.
3. Introduce tu usuario y **contraseña de Windows** en la ventana de credenciales. El PIN de Windows Hello no sustituye esa contraseña.
4. Comprueba que el instalador muestre las tres tareas registradas. Si hay un error, revisa cuáles quedaron instaladas; no se afirma que la instalación haya terminado.
5. Abre el Programador de tareas y localiza GBM_Forward_Collector, GBM_Forward_Resolver, GBM_Forward_Catchup y GBM_Backup_Daily. Usa “Ejecutar” para una comprobación y consulta el log propio de cada trabajo.

**El instalador está entregado, no ejecutado en modo de registro.** Se generaron los XML en modo Preview y el componente de Windows validó su esquema. Ninguna tarea quedó instalada durante el desarrollo.

No se guardan contraseñas en archivos del proyecto, logs ni argumentos del proceso. El instalador entrega la credencial en memoria al servicio de Windows; este administra la credencial del trabajo. Las tareas corren con privilegios mínimos, no con una consola de trading abierta. Si cambia la contraseña de Windows, actualiza la credencial de las tareas.

Alternativa sin proporcionar contraseña, ejecutando desde PowerShell:

~~~powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_autopilot_tasks.ps1
~~~

Esta alternativa requiere que el usuario haya iniciado sesión en Windows. La tarea de recuperación usa inicio de sesión en vez de arranque. No equivale al modo verdaderamente desatendido.

Para inspeccionar sin instalar:

~~~powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_autopilot_tasks.ps1 -Preview -Unattended
~~~

Para actualizar **solo estas tres tareas**, tras revisar sus nombres:

~~~powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\install_autopilot_tasks.ps1 -Unattended -Replace
~~~

En Windows PowerShell, para pasar la lista correctamente desde una sesión PowerShell también puedes usar:

~~~powershell
& .\scripts\install_autopilot_tasks.ps1 -Unattended -Replace -Symbols @("SMCI", "NVDA")
~~~

## Trabajos

| Tarea | Hora principal NY | Script | Función |
|---|---|---|---|
| GBM_Forward_Collector | L–V 11:00 | daily_auto_collector.py | Calcula y registra las seis zonas disponibles por activo |
| GBM_Forward_Resolver | L–V 17:00 | auto_resolver.py | Resuelve todos los registros vencidos |
| GBM_Forward_Catchup | L–V 09:05 + arranque con demora 5 min | boot_catchup.py | Recupera pendientes, incluso de varios días atrás |
| GBM_Backup_Daily | L–V 18:00 | github_backup.py --encrypt | Crea instantánea SQLite consistente y cifrada; si la PC está apagada, StartWhenAvailable la recupera al siguiente inicio |

Activos iniciales: **SMCI y NVDA**. Los resolutores también recuperan símbolos retirados de la lista si aún tienen evidencia pendiente.

El horario es America/New_York, no la zona local de México. Windows recibe dos candidatos UTC por tarea para cubrir EST/EDT; el colector y el resolver validan la hora NY y el candidato incorrecto termina sin operar. Catch-up admite una ventana más amplia (09:05–17:20) para arranques tardíos: el segundo candidato puede realizar otra comprobación, pero no vuelve a resolver registros cerrados.

No cambia el reloj ni la zona horaria de Windows. La UI del Programador puede mostrar horas locales distintas; los logs indican explícitamente EST/EDT.

- Colector programado: permite emisión entre 11:00 y antes de 11:20 NY para tolerar demoras/reintentos.
- Resolver programado: entre 17:00 y antes de 17:20 NY, respetando además cierre bursátil + 15 minutos.
- Catch-up: puede recuperar sesiones anteriores en festivos; nunca crea predicciones retrospectivas.
- Fines de semana/festivos: colector y resolver terminan normalmente sin trabajo.
- Calendario XNYS reconoce cierres anticipados y horario de verano.
- Reintentos: hasta tres, cada cinco minutos; duración máxima de cada tarea: 15 minutos.
- StartWhenAvailable recupera disparos perdidos. Si el colector llega fuera de su ventana, no simula que predijo a las 11:00. Ese día puede quedar sin pronósticos.
- IgnoreNew y un bloqueo de archivo entre procesos evitan solapamientos. El bloqueo se libera al terminar o morir el proceso.
- No se exige conexión como condición de Windows: el script registra el fallo de internet y devuelve código 1 para que el programador reintente.
- El PC **debe estar encendido**. Las tareas no lo encienden; WakeToRun está desactivado. Si se apaga antes de resolver, el siguiente arranque recupera las sesiones pendientes.

## Ejecución manual

Desde la raíz del proyecto:

~~~powershell
.\.venv\Scripts\python.exe scripts\daily_auto_collector.py --symbols SMCI NVDA
.\.venv\Scripts\python.exe scripts\auto_resolver.py --symbols SMCI NVDA
.\.venv\Scripts\python.exe scripts\boot_catchup.py --symbols SMCI NVDA
~~~

Sin --scheduled, el colector manual puede funcionar durante la sesión, no solo a las 11:00. La fecha guardada siempre es el momento real de emisión. Repetir una colección cuando ya hay un corte completo del día no duplica las seis zonas; después de las 11:00 se exige que ese corte no sea anterior a las 11:00 NY.

Diagnóstico sin descargas ni escrituras en DB:

~~~powershell
.\.venv\Scripts\python.exe scripts\daily_auto_collector.py --check-only
~~~

Código de salida 0: completado o no había trabajo autorizado. Código 1: error recuperable/integridad/muestra incompleta; revisar el log. No se muestran ventanas gráficas en las tareas porque usan pythonw.exe.

## Fidelidad al motor y límites

No se editó app.py, el cálculo de probabilidades ni el código de los motores.

La ruta headless reutiliza:
- parámetros aprobados por activo/versión;
- analyze_probability con require_fresh=True y memoria macro;
- el mismo filtro fundamental y fallback firmado;
- synchronize_position, para conservar el plan fijado de las posiciones reales;
- build_zone_snapshot y log_snapshot, exactamente los adaptadores que usa el flujo UI/PDF;
- el mismo resolutor firmado de forward testing.

No genera PDFs ni gráficos; no importa Streamlit. La calibración multiclase de horizontes no se repite porque **no alimenta los niveles ni las probabilidades de toque/cierre de las seis zonas**. No se omiten el filtro fundamental ni el plan persistente que sí afectan esos niveles.

Como en la UI, se pueden añadir eventos analíticos en operational_events y cortes en fundamental_news_snapshots. **No se añaden órdenes, compras, ventas ni movimientos de efectivo.** No se ejecuta Database.initialize ni se fuerza una migración del libro.

Datos:
- Primera descarga 5m: 1 mes, igual que la UI.
- Siguientes: desde la última descarga con un día de solape; busca huecos anteriores dentro de la ventana.
- Una revisión grande de precios en el solape fuerza recarga del mes para no mezclar bases de precio.
- Descarga diaria: cinco años, como el análisis principal de la UI, con caché por fecha NY.
- Solo velas cerradas. Un archivo de caché con firma inválida no se utiliza.
- Se guardan adquisiciones con SHA-256 en data/autopilot/market, más una caché actual atómica.
- SQLite ya evita duplicar el mismo corte/zona/modelo; una interrupción después de guardar se reconoce al reiniciar.
- Datos N/D no se convierten en 0% ni se inventan soportes para cumplir artificialmente “seis”. La ejecución incompleta se registra y devuelve 1.
- Un fallo de un activo no impide procesar el siguiente.
- Resolver usa precios históricos de cada sesión; nunca la apertura de hoy para resolver ayer.
- Algunos toques pueden quedar ambiguos aunque el cierre ya esté resuelto. No se falsifica actual_touch_occurred para borrar los NULL.

Los archivos **logs/collector.log**, **logs/resolver.log** y **logs/catchup.log** se rotan a 2 MB, con cinco copias cada uno. Logs y datos locales están ignorados por Git. Ni los registros forward ni los archivos de caché son prueba de calibración por sí solos.

## Comprobación realizada con datos reales

3 de septiembre de 2026, aproximadamente 11:49 NY:
- SMCI: ya existía un corte completo de hoy; no duplicado.
- NVDA: seis predicciones nuevas guardadas con precios/estimaciones actuales.
- Segunda ejecución: ambos reconocidos, cero duplicados.
- Resolver a las 11:50: omitido correctamente, todavía no había cerrado la sesión.
- Catch-up: no resolvió prematuramente los registros del día.
- Hashes antes/después iguales en cash_movements, trades, receipts, fx_rates, portfolio_snapshots, schema_migrations y settings.
- Evidencia local de comparación: data/autopilot/acceptance_collection.json.

La resolución real de estos pronósticos **queda pendiente del cierre**. Se probaron resolución y recuperación con bases aisladas, sin afirmar que se haya observado un cierre que aún no ocurrió.

## Si falla

- Revisa el log y “Último resultado de ejecución” en Windows.
- Si se movió el proyecto, reinstala con -Replace para corregir las rutas.
- Si no hay contraseña válida para ejecución desatendida, usa modo interactivo y mantén la sesión iniciada.
- No copies ni envíes tu contraseña aquí.
- Si faltan velas antiguas porque el proveedor ya no las conserva, la resolución queda pendiente; el sistema no rellena resultados inventados.
- Si deseas cambiar la fórmula o extraer en el futuro toda la orquestación UI/headless a un servicio común, hazlo como tarea separada. Esta implementación mantiene intacto el motor solicitado.

Referencias oficiales: [horario StartBoundary de Windows](https://learn.microsoft.com/en-us/windows/win32/api/taskschd/nf-taskschd-itrigger-put_startboundary), [opciones del programador](https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/new-scheduledtasksettingsset).
