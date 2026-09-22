# Walk-forward anidado, embargo XNYS y holdout final sellado

Este protocolo valida por separado cada combinación `símbolo + horizonte` y
usa como objetivo operacional `TP_FIRST / SL_FIRST / TIMEOUT`. No mezcla
observaciones, hiperparámetros ni resultados entre 1 hora, 6 horas, 1 día, 1
semana, 1 mes y 6 meses.

## Estructura de validación

1. Se reserva cronológicamente el 20% final como holdout sellado, con un mínimo
   de 60 observaciones.
2. Antes del holdout se aplica un embargo en sesiones XNYS y se purga cualquier
   fila cuyo resultado todavía no era conocido al comenzar el holdout.
3. Sobre el bloque de desarrollo se ejecutan folds externos expansivos. Cada
   fold externo representa una prueba walk-forward todavía no observada por su
   entrenamiento.
4. Dentro de cada fold externo se ejecutan folds internos expansivos. La
   penalización L2 se selecciona exclusivamente por log-loss interno.
5. Cada fold vuelve a ajustar imputación, media y escala únicamente con su
   entrenamiento. No reutiliza estadísticas calculadas con validación.
6. El agregado externo debe mejorar simultáneamente el Brier y el log-loss del
   prior de clases, y al menos la mitad de los folds debe mostrar habilidad.
7. Solo entonces se intenta calibrar. Se reúnen las predicciones externas
   fuera de muestra, se ajusta una temperatura multiclase con la parte
   cronológicamente temprana y se comprueba en la parte posterior. Entre ambas
   partes se aplican el mismo embargo XNYS y la purga de etiquetas tardías.
   La temperatura se elige únicamente con la parte temprana.
8. El modelo crudo ya debe mejorar el baseline en la validación posterior. La
   calibración también debe mejorar simultáneamente Brier y log-loss tanto
   frente al score crudo como frente al baseline. Si falla, queda
   `REJECTED_CALIBRATION_FINAL_HOLDOUT_UNOPENED`.
9. Solo entonces se congela el protocolo: features, L2, folds, calibrador,
   métricas de desarrollo y compromiso SHA-256 del holdout.
10. El holdout se abre contra ese hash y compara score crudo, calibrado y
    baseline una vez. El resultado calibrado solo se aprueba si supera los
    otros dos en Brier y log-loss; en caso contrario no se publica modelo.

Un modelo aprobado declara `HISTORICAL_OOS_CALIBRATED_PRELIMINARY`: la
calibración procede del replay histórico y todavía requiere verificación
forward real antes de atribuirle fiabilidad operativa. Los seis resultados
permanecen separados por activo y horizonte.

## Embargo dependiente del horizonte

| Horizonte | Embargo mínimo |
|---|---:|
| 1 Hora | 1 sesión XNYS |
| 6 Horas | 1 sesión XNYS |
| 1 Día | 1 sesión XNYS |
| 1 Semana | 5 sesiones XNYS |
| 1 Mes | 21 sesiones XNYS |
| 6 Meses | 126 sesiones XNYS |

El embargo usa el calendario bursátil, por lo que no cuenta fines de semana o
festivos como sesiones. La purga por `label_available_at` se aplica además del
embargo: ambas protecciones cumplen funciones distintas.

## Holdout sellado

El conjunto final se compromete mediante SHA-256 sobre identificador de corte,
timestamps, hash de features y resultado. Antes de abrirlo se genera
`protocol_frozen_sha256`. Las métricas finales registran expresamente el hash
del protocolo contra el que fueron abiertas. Un artefacto con métricas en un
holdout `SEALED_UNOPENED` es inválido.

Mientras permanece sellado, tampoco se publica su distribución de clases.
La versión anterior del contrato (`V1`) sigue siendo verificable por integridad;
la calibración nueva se registra bajo `NESTED_EMBARGOED_WALK_FORWARD_V2`.

Si el desarrollo no supera el baseline, el artefacto conserva solamente el
compromiso del holdout, sin Brier, log-loss, accuracy ni coeficientes derivados
de ese bloque.

## Estado real actual

Los artefactos disponibles contienen entre 18 y 19 resultados resueltos por
horizonte. El mínimo preregistrado es 300, antes de descontar embargo y purgas.
Por ello SMCI y NVDA quedan como
`INSUFFICIENT_DATA_FINAL_HOLDOUT_UNOPENED`. No se abrió el holdout, no se
calcularon métricas finales y ningún modelo fue promovido.

## Ejecución

```powershell
.\.venv\Scripts\python.exe scripts\run_nested_walk_forward.py
```

Los artefactos firmados se escriben en `output/nested_walk_forward/`. La rutina
es headless, no escribe SQLite y no modifica efectivo, comprobantes, posiciones
u órdenes.
