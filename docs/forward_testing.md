# Forward testing de zonas · contrato v1
Fecha de instalación: 2 de septiembre de 2026.

## Estado real al instalar

**PRELIMINAR. No existe todavía evidencia prospectiva para afirmar que 71% acierta el 71%.**

- Histórico real Yahoo/yfinance descargado: SMCI, 670 sesiones diarias, 2024-01-02 a 2026-09-02.
- Comprobación con datos reales de 2026-09-02: 78 de 78 velas 5m esperadas; último cierre 5m y cierre diario: USD 37.00.
- Contexto calculado solo con periodos cerrados: 670 diarios, 139 semanales, 32 mensuales. EMA200 semanal y EMA50/200 mensual: N/D, no hay periodos suficientes.
- Registro prospectivo al instalar: **0 pronósticos, 0 resultados**. Instalación después del cierre: no se reconstruyeron predicciones a posteriori.
- Se añadieron exclusivamente tres tablas analíticas. Se compararon hashes de contenido y esquema de todas las tablas preexistentes antes/después: sin cambios.
- No se ejecutaron migraciones antiguas, no se sembró efectivo y no se alteraron órdenes, recibos, cifrado o respaldos.

Estos datos prueban disponibilidad del proveedor y coherencia de cierre, **no precisión predictiva**.

## Uso

1. Reiniciar la aplicación con el entorno habitual.
2. Abrir Motor cuantitativo durante una sesión de bolsa y generar el análisis. El snapshot que alimenta los cuatro PDF y la interfaz registra ENTRY1/2/3, TP1, TP2 y R3 si existen nivel y estimación válidos.
3. Consultar la nueva pestaña **VALIDACIÓN**. Puede abrirse aunque falle el análisis de mercado actual.
4. La resolución se intenta al abrir/refrescar el Motor (máximo una consulta por 15 minutos), para todas las sesiones pendientes ya vencidas.
5. También ejecutar manualmente desde la carpeta del proyecto:

```powershell
.\.venv\Scripts\python.exe scripts\resolve_daily_predictions.py
```

Actualizar/verificar el histórico diario:

```powershell
.\.venv\Scripts\python.exe scripts\resolve_daily_predictions.py --download-daily SMCI
```

**No se instaló un scheduler del sistema operativo.** Con la aplicación cerrada no hay un proceso autónomo recolectando pronósticos. El script resuelve registros ya existentes, no genera señales nuevas. Si se configura un programador, ejecutar tras el cierre real + 15 minutos; el script igualmente valida el calendario. Reabrir el Motor al día siguiente recupera pendientes, mientras el proveedor conserve sus velas.

## Persistencia y contrato

- `zone_prediction_log`: UUID, emisión UTC real, corte de la última vela cerrada, sesión/vencimiento, símbolo, identidad y límites de zona, precio de referencia, probabilidades originales en [0,1], contexto, SHA del modelo, firma inicial y resolución.
- `zone_market_evidence`: OHLCV del proveedor usado para resolver, fecha de descarga y SHA-256.
- `zone_daily_validation`: resúmenes diarios versionados por membresía de resultados, símbolo y modelo; Brier separado para toque y cierre.
- Inicialización aditiva e idempotente independiente del libro contable. No modifica su versión de migración.
- SQL bloquea editar pronósticos, volver a resolver o borrar registros. Se verifican firmas de pronóstico, resolución y evidencia antes de computar métricas.
- SHA-256 detecta alteraciones; no protege frente a un administrador capaz de reemplazar datos, código y firmas. Se mantienen las protecciones de respaldo existentes.
- Identidad única: símbolo + versión del modelo + vela de origen + zona. La primera emisión gana; refrescar o descargar cuatro PDF no cuadruplica observaciones.
- Cambiar el código analítico produce otra versión. Reiniciar tras desplegar cambios. Los resultados de distintas versiones no se mezclan.
- Un PDF generado externamente por una función pura no escribe en la DB: la integración persistente está en el flujo UI/PDF de la aplicación.
- No se inventa probabilidad de STOP; el esquema admite STOP para cuando el modelo emita ese evento explícitamente.

## Eventos y causalidad

**Toque:** alcanzar o sobrepasar el borde cercano de la zona desde el precio de referencia. Si el precio está arriba, Low <= borde superior; si está abajo, High >= borde inferior. Un gap que salta el nivel cuenta como alcance del umbral, **no como ejecución garantizada a ese precio**.

**Cierre:** compras: Close <= límite inferior; ventas: Close >= límite superior. La relación descriptiva ABOVE/BELOW/INSIDE usa ambos límites, incluyendo igualdad en INSIDE. No se inventa P(cierre arriba)=1-P(cierre abajo) para zonas con interior. Se conserva también la probabilidad direccional de cierre.

Solo cuenta movimiento posterior a la emisión. Las velas 5m no permiten saber en qué segundo ocurrió un extremo:
- Un cruce en una vela íntegramente posterior confirma el evento.
- Si solo cruza la vela parcialmente anterior, toque ambiguo y excluido; el cierre sí es evaluable.
- Si la zona ya estaba alcanzada al corte, el toque se excluye de evaluación predictiva.
- No se registra fuera de sesión, con corte vencido, timestamp futuro/retrospectivo o probabilidad N/D.
- Se exigen las velas 5m completas de la sesión, OHLCV finito y coincidencia del cierre diario con la última vela 5m. Discrepancias quedan pendientes; nunca se sustituyen por el precio del día siguiente.
- Calendario XNYS: festivos, horario de verano y cierres anticipados. No se fija UTC-4 durante todo el año.

## Métricas e interpretación

- Auditoría conserva todas las emisiones; cohorte principal: **primera emisión por sesión/zona/versión**, elegida antes de conocer el resultado.
- Si la primera queda ambigua, pendiente o corrupta, no se reemplaza por una posterior favorable.
- Brier binario = promedio de (p - resultado)^2. UI acumulada: promedio de los Brier diarios, con peso igual por sesión.
- Toque y cierre separados; filtros por símbolo, versión y nivel.
- Curva exploratoria en intervalos predefinidos de 10 pp: pronóstico medio vs frecuencia real, con cantidad de observaciones, eventos y sesiones.
- Seis niveles del mismo activo/día están correlacionados. No tratar 60 sesiones x 6 niveles como 360 ensayos independientes.
- Un Brier bajo o una frecuencia compatible no demuestra calibración. Hay que evaluar incertidumbre por bloques de sesión, estabilidad por régimen/activo y comparación contra una referencia, sin ajustar sobre la misma muestra utilizada para verificar.
- La curva actual **no es una curva OOS validada del modelo nuevo**: es seguimiento prospectivo descriptivo. No se ajusta isotónica ni se modifica el motor a partir de estos resultados.
- 90 días calendario equivalen aproximadamente a 60 sesiones; 90 días hábiles bursátiles son 90 sesiones. Ningún umbral temporal garantiza demostrar exactitud del 71%.
- No se aplicará “42 de 60 => calibrado”. La frecuencia por sí sola no prueba igualdad con 71% y la incertidumbre/dependencia debe examinarse.

## Archivos y datos

- `portfolio_tracker/services/zone_forward.py`: modelo, integridad, almacenamiento, resolución y cohortes.
- `portfolio_tracker/repository.py`: API save_prediction, resolve_predictions, zone_predictions.
- `portfolio_tracker/services/forward_market.py`: proveedor, caché y contexto cerrado.
- `portfolio_tracker/services/market_data.py`: entrada pública download_daily_history.
- `portfolio_tracker/ui/forward_validation.py`: panel y actualización acotada.
- `scripts/resolve_daily_predictions.py`: operación manual/catch-up y verificación de tablas preexistentes.
- `tests/test_zone_forward.py`: datos sintéticos exclusivamente en bases temporales; validan el software, no rentabilidad.
- Caché privada local: `data/forward_market/SMCI_daily_2024.json`, firma y fecha de adquisición incluidas.
- No cambian las dependencias ni se requieren credenciales nuevas.

La retención intradía del proveedor es limitada: resolver y archivar con regularidad. Referencia primaria: https://ranaroussi.github.io/yfinance/reference/yfinance.functions.html

