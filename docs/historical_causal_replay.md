# Replay causal histórico · cortes 11:00 NY

El replay histórico reconstruye qué conocía el motor en cada sesión XNYS a las
11:00 de Nueva York. Su contrato es
`XNYS_1100_HISTORICAL_CAUSAL_REPLAY_V1` y queda separado de las observaciones
forward reales.

## Contrato temporal

Para cada sesión disponible:

1. El analizador recibe solamente velas 5m cerradas hasta el corte. Las velas
   posteriores se excluyen antes de calcular indicadores, régimen, setup,
   gatillo, niveles y scores.
2. Se construye un único snapshot de features, versiones y hashes de código.
   Los seis horizontes comparten ese snapshot y congelan sus predicciones y
   contratos.
3. Después del pronóstico, el etiquetador puede leer el tramo futuro para
   determinar el cierre exacto `UP / RANGE / DOWN` y el primer evento
   `TP_FIRST / SL_FIRST / TIMEOUT`.
4. Si el histórico termina antes del vencimiento, el resultado queda
   `RIGHT_CENSORED`. Si falta una trayectoria causal continua o el cierre
   exacto, queda `MISSING_CAUSAL_PATH` o `MISSING_EXACT_CLOSE`; nunca se rellena
   con la última cotización disponible.

Un cambio en precios posteriores puede cambiar la etiqueta, pero no las
features ni la predicción congelada. Esto se prueba expresamente en
`tests/test_historical_replay.py`.

## Separación frente a OOS real

Los artefactos se escriben en `output/historical_replay/*.json`. No se insertan
en `live_model_observations`, `operational_model_outcomes` ni en tablas
contables. Cada manifiesto declara
`REPLAY_NEVER_COUNTS_AS_LIVE_OOS`: sirve para investigación, entrenamiento y
walk-forward, pero no demuestra calibración forward real.

Cada corte y el contenido determinista completo llevan SHA-256. Modificar una
feature, una predicción o un resultado invalida la verificación. El replay
también registra explícitamente estas limitaciones:

- no reconstruye noticias o fundamentales históricos que no tengan snapshot
  punto-en-tiempo;
- no usa posiciones, efectivo ni comprobantes actuales;
- la correlación SMCI/NVDA solo se incorpora si ambos históricos firmados están
  disponibles en el mismo corte.

## Ejecución

La caché firmada del autopiloto puede utilizarse directamente:

```powershell
.\.venv\Scripts\python.exe scripts\build_historical_replay.py `
  --symbols SMCI NVDA --max-cuts 20
```

Para acotar fechas:

```powershell
.\.venv\Scripts\python.exe scripts\build_historical_replay.py `
  --symbols SMCI NVDA --start 2026-09-01 --end 2026-09-18
```

El proveedor público conserva una ventana 5m limitada. Construir cientos o
miles de cortes exige suministrar un histórico intradía licenciado y ajustado
punto-en-tiempo; el motor no fabrica sesiones antiguas a partir de velas
diarias.
