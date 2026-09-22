# Modelos regularizados separados por horizonte

El entrenador consume exclusivamente artefactos verificados del replay causal
histórico y genera un clasificador multinomial L2 distinto para cada par
`símbolo + horizonte`:

- 1 Hora
- 6 Horas
- 1 Día
- 1 Semana
- 1 Mes
- 6 Meses

El objetivo no es «sube o baja al cierre». Es el primer evento operacional ya
definido por el motor: `TP_FIRST`, `SL_FIRST` o `TIMEOUT`. Ningún modelo mezcla
filas, parámetros ni métricas de otro horizonte.

## Protocolo causal

1. Se usan solamente features congeladas al emitir la predicción. Los precios
   futuros y el resultado nunca entran al vector de entrada.
2. Cada horizonte se divide cronológicamente 60% entrenamiento, 20%
   calibración/selección y 20% holdout.
3. Una fila de entrenamiento cuya etiqueta todavía no era conocida al iniciar
   calibración se purga. Se aplica lo mismo entre calibración y holdout.
4. Medianas, medias y desviaciones se ajustan únicamente con entrenamiento.
5. La penalización L2 se elige únicamente por log-loss en calibración.
6. El modelo se reajusta con entrenamiento + calibración congelada y se evalúa
   una sola vez en holdout. Brier, log-loss y accuracy publicados pertenecen
   exclusivamente a ese holdout.
7. Se compara contra el prior de clases aprendido sin holdout. Si no mejora
   simultáneamente Brier y log-loss, queda `REJECTED_NO_OOS_SKILL`.

La salida softmax se etiqueta `REGULARIZED_SCORE_UNCALIBRATED`: entrenar un
clasificador no demuestra calibración. La calibración empírica y el forward
live siguen siendo etapas separadas.

## Umbral profesional y estado actual

El valor predeterminado exige al menos 300 observaciones resueltas por símbolo
y horizonte, al menos 20 ejemplos de cada clase en entrenamiento y 60 filas en
holdout. Con menos evidencia el proceso no fuerza un ajuste: produce un
artefacto firmado `INSUFFICIENT_DATA`, sin coeficientes y no promovible.

Los históricos intradía firmados disponibles actualmente contienen solo unas
decenas de cortes para SMCI y NVDA. Por ello los doce candidatos reales quedan
rechazados por muestra insuficiente. Esto es el comportamiento correcto: no se
presentan como modelos listos para operar ni se conectan al motor vivo.

## Ejecución

Primero se construye el replay y luego se entrena:

```powershell
.\.venv\Scripts\python.exe scripts\build_historical_replay.py --symbols SMCI NVDA
.\.venv\Scripts\python.exe scripts\train_horizon_models.py
```

Los resultados se guardan en `output/horizon_models/*.json`. Cada modelo y el
artefacto completo llevan SHA-256. La rutina no abre Streamlit, no usa SQLite y
no modifica contabilidad, posiciones ni órdenes.

Para alcanzar 300+ muestras reales se necesita un histórico intradía 5m
licenciado, ajustado y punto-en-tiempo. La ventana pública disponible no debe
rellenarse fabricando velas o reutilizando datos diarios.
