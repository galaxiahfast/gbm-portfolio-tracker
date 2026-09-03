# Motor operativo con memoria

## Contratos

- `select_last_closed_bar(frame, timeframe, as_of)`: conserva únicamente barras
  cuyo cierre de calendario ya ocurrió. Etiquetas OHLCV de inicio, intradía
  naive interpretado en Nueva York; fechas diarias como etiquetas de sesión.
- `resample_closed`: 1h y 4h ancladas a 09:30 Nueva York, sin mezclar sesiones,
  con control de cobertura 5m. El último bucket puede ser más corto al cierre.
- `RegimeEngine`: semanal/diario/4h, sin entrada 5m. Exige concordancia para
  LONG_ONLY/SHORT_ONLY. Falta de historia → NO_TRADE; desacuerdo/rango →
  BOTH_REDUCED, que en esta versión no autoriza entradas inmediatas.
- `SetupEngine`: estructura 4h/1h, soporte/resistencia y permisos direccionales.
- `TriggerEngine`: cruce estocástico, MACD, ruptura de vela previa y volumen
  superior a la media de las 20 velas anteriores. Solo recibe velas cerradas.
- `analyze_probability(..., as_of_time=..., previous_macro_trending=...)`:
  cálculo reproducible; replay histórico debe suministrar el corte real, no
  usar la hora actual. Los scores diagnósticos conservan su naturaleza heurística.

## Memoria y contabilidad

Migración 8 agrega únicamente `operational_events`, con snapshots JSON y cadena
SHA-256. No cambia trades, efectivo, comprobantes ni algoritmos de respaldo AES.
`synchronize_position(database, analysis)` usa una transacción SQLite para leer
inventario real y actualizar solo esta memoria. Reruns idénticos no duplican eventos.

Estados: FLAT → LONG_ACTIVE/SHORT_ACTIVE por inventario confirmado, nunca por
un score. La contabilidad existente no habilita ventas en corto por esta mejora;
SHORT_ONLY es permiso analítico, no permiso del broker ni inventario ficticio.
En una posición activa se fijan dirección, stop y objetivos. Stop, TP1 o pérdida
de estructura 4h → EXIT_PENDING. Un TP1 solicita revisar una salida parcial;
no presume que se ejecutó. El estado vuelve a FLAT solo al cerrarse el inventario
en el libro. Si stop y TP se tocan en un mismo intervalo, prima la alerta de stop.

Posiciones preexistentes se adoptan con un plan al primer corte disponible. Se
indica explícitamente que este no es el plan original de entrada. No se modifican
el costo ni las operaciones históricas. Una salida parcial conserva el plan;
una nueva entrada tras quedar flat inicia un nuevo episodio.

ADX diario: se activa >25, se mantiene >20, se desactiva <=20. Se restaura desde
SQLite; sin memoria se reproduce la histéresis con la historia disponible. ADX5m
no redefine el permiso macro. Los filtros previos de riesgo permanecen como vetos
adicionales; la calibración no puede saltarse un gatillo pendiente.

## Seguridad y límites

La auditoría verifica la cadena operacional; una discrepancia bloquea el motor.
SHA-256 detecta alteraciones respecto a la cadena almacenada, no autentica frente
a un atacante que reescriba la base completa. No se envían órdenes al broker.
Los stops son alertas calculadas al cierre de barras, no protección en tiempo real
intrabar: deben existir órdenes de protección reales si se requiere esa cobertura.
La memoria se actualiza cuando se consulta el predictor; no es un servicio daemon.

El calendario XNYS contempla DST, festivos y cierres anticipados. Datos fuera de
sesión, buckets incompletos y velas abiertas no participan en decisiones. Un
proveedor con datos erróneos o retrasados sigue requiriendo controles operativos.

## Comprobación

Instalar `requirements.txt` (se agrega exchange-calendars) y ejecutar
`python -m pytest -q`. La base productiva migra al iniciar normalmente la app;
las pruebas usan bases temporales. Los dos paneles de niveles son escenarios,
no dos órdenes simultáneas. La decisión del PDF reconoce la posición activa.
