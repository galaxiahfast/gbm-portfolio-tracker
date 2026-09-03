# F05/F06 — Simetría y replay técnico causal

Fecha: 2026-09-02. Motor: `causal-replay-phase5-v2`.

## F05: orientación única del score

El score de investigación diaria parte de 3 puntos y proyecta cada voto firmado
(MACD, precio/EMA, contexto semanal y mensual) sobre LONG (+1) o SHORT (-1)
exactamente una vez. El bono por volumen es independiente del lado. Un mercado
espejo recibe el mismo score, bucket y exposición. Los votos neutrales valen cero.
Esto no elimina los costes reales distintos que puedan existir al vender corto.

## F06: un núcleo, dos consumidores

`analytics/causal_core.py` contiene RegimeEngine, SetupEngine, TriggerEngine,
histéresis y `evaluate_causal_core`. `decision_engines.py` conserva los imports
anteriores como adaptador de compatibilidad, sin duplicar las reglas.

El motor técnico en vivo y `analytics/replay.py` llaman al mismo
`analyze_probability(..., as_of_time=...)` y al mismo núcleo final de autorización.
Se conservan los filtros técnicos adicionales, vetos, ATR y TP1 del analizador.
Una condición previa barata, compartida con TriggerEngine, descarta únicamente
velas que no pueden tener gatillo; no puede autorizar entradas.

`ReplayDataset(intraday, daily, as_of)` exige OHLCV real de 5 minutos y contexto
diario. Cada corte descarta velas en formación 5m/1h/4h/diarias. La memoria ADX
se reconstruye desde el prefijo diario cerrado, igual que un inicio en frío en
vivo. Los cachés son locales al dataset/ATR, sin guardar objetos de análisis
pesados ni mezclar ejecuciones o activos.

La entrada simulada usa la apertura de la siguiente vela, con stop y TP1
congelados del analizador. Un gap que invalide el plan o su R:R cancela la entrada.
Los stops con gap se ejecutan conservadoramente a la apertura adversa; si una
vela toca stop y TP, prevalece el stop. Las posiciones no se solapan. El vencimiento
cuenta sesiones de Nueva York; el embargo de entrenamiento usa 78 velas por
sesión (conservador ante sesiones cortas). Solo resultados cuyo cierre precede
el comienzo de la siguiente ventana alimentan su calibración.

## Selección explícita, sin configuración global

`repository.latest_backtest_parameters(symbol=..., engine_version=...)` requiere
ambos argumentos. Solo devuelve configuraciones de corridas APPROVED cuyo activo
también fue aprobado individualmente. Verifica el SHA-256 del payload, contrato
de replay, versión y huella del código causal, dataset y coherencia entre el JSON
firmado y los metadatos de parámetros/activos. Reevalúa los mínimos de capital.
Si no hay evidencia compatible devuelve None y la interfaz usa valores base.

Las corridas diarias siguen disponibles para investigación y pruebas históricas,
etiquetadas `daily-research-symmetric-v2`, pero quedan rechazadas para promoción
al motor intradía. Los registros existentes no se borran ni se reescriben.

## Alcance y límites verificables

- La fuente actual de la interfaz entrega un mes 5m y cinco años diarios. El
  contexto diario no aumenta la muestra intradía. Para cientos de señales OOS
  hace falta suministrar a ReplayDataset un archivo 5m histórico más amplio.
- La paridad es del motor **técnico** y su núcleo de entradas. No se reproducen
  titulares/fundamentales actuales sobre fechas pasadas. Tampoco se afirma
  equivalencia con fills reales del broker, spreads/borrow fees o la reconciliación
  de posiciones contables. La simulación usa stop/TP1/vencimiento; no replica
  aún salidas parciales ni todos los eventos persistentes de gestión estructural.
- Un replay idéntico al código no demuestra rentabilidad. Se mantienen los
  mínimos OOS, límites de riesgo y reporte de costes; no se aprueban muestras vacías.
- No hay nueva migración ni cambios en tablas/registros contables. Las pruebas
  solo crean bases temporales. Reiniciar Streamlit carga los módulos actualizados.
