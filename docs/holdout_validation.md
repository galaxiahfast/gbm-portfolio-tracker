# Validación final de modelos operativos

El holdout final se reserva **antes de leer sus resultados** en un registro SQLite
separado de la contabilidad: `data/model_validation/holdout_openings.sqlite`.
Cada apertura registra SHA-256 del dataset, SHA-256 del protocolo congelado,
SHA-256 del compromiso del holdout, símbolo, horizonte y timestamp UTC. La
restricción única por dataset/símbolo/horizonte impide repetir la corrida; la
restricción por `cut_id` también impide reutilizar una parte del holdout después
de exportar un dataset ligeramente diferente. Una segunda apertura lanza
`HoldoutAlreadyOpenedError`. No se borra ni reinicia ese registro para volver a
optimizar sobre las mismas observaciones.

La calibración requiere al menos cinco ejemplos de cada clase (`TP_FIRST`,
`SL_FIRST`, `TIMEOUT`) **tanto en ajuste OOF como en validación OOF**. El holdout
final exige el mismo soporte por clase para aprobar. Si falta alguna clase, el
modelo queda rechazado, aunque sus promedios de Brier parezcan mejorar.

Las métricas Brier, log-loss y exactitud muestran intervalos del 95% calculados
con bootstrap de bloques móviles en orden temporal; la proporción de folds
exitosos usa Wilson. Son intervalos de incertidumbre muestral, no promesas de
rentabilidad ni corrección por selección múltiple. No se usan para escoger los
hiperparámetros ni para volver a abrir el holdout.

## Mínimo por horizonte

El piso de 300 muestras elegibles se conserva, pero para cada horizonte se
incrementa según su embargo `E` en sesiones NYSE. El primer entrenamiento
externo debe sobrevivir a `E` y todavía permitir el entrenamiento/validación
internos. También se reserva capacidad para todos los folds externos y para
ajuste/validación OOF cronológicos. Si `D` es ese desarrollo mínimo, se busca
el primer `N >= 300` tal que:

`N - max(holdout_mínimo, ceil(N × fracción_holdout)) - E >= D`

El código de `minimum_samples_for_horizon()` documenta el cálculo de `D`.
Esta cota conservadora presupone como máximo un corte por sesión; si existen
múltiples cortes, el purgado real por calendario y disponibilidad de etiquetas
se verifica igualmente en cada fold. Seis meses requieren más que 300 muestras
debido al embargo de 126 sesiones; el motor devuelve evidencia insuficiente
en vez de realizar un entrenamiento inviable o relajar el embargo.
