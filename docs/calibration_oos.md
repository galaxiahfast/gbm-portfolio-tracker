# F03 / F04 — Contrato de calibración cronológica

Fecha: 2026-09-02.

## Particiones y causalidad

La división nominal, ordenada por emisión (`observed_at`), es 60% entrenamiento,
20% calibración y 20% prueba/holdout. No se barajan observaciones. Se purgan
resultados conocidos (`resolved_at`) después de comenzar el bloque siguiente,
y horizontes solapados dentro de cada bloque. Los tamaños efectivos pueden
ser menores que 60/20/20; se muestran en la interfaz junto al total purgado.

El primer bloque se reserva al modelo base y calcula las frecuencias de clase
del benchmark. Este cambio NO entrena retrospectivamente el motor de indicadores.
La isotónica se ajusta exclusivamente en el segundo bloque. Brier, Brier crudo,
Brier del benchmark y reliability curves se calculan exclusivamente en holdout.
Cambiar etiquetas del holdout no puede alterar la curva ni la predicción actual.

## Tres clases, un único predictor

Al emitir cada horizonte se congelan los límites de rango y el vector original:
subida = cierre superior al techo; rango = cierre dentro de límites inclusivos;
bajada = cierre inferior al piso. Son eventos de **cierre al vencimiento**, no
probabilidades de tocar zonas durante la sesión.

El predictor conjunto usa tres curvas isotónicas one-vs-rest y normalización
al simplex dentro del modelo. La regularización fija de los puntos se aplica
antes de PAV, preservando monotonía. Se evalúa el vector normalizado completo;
la UI no vuelve a recortar ni redistribuir sus componentes. Brier multiclase
se define como media de la suma de tres errores cuadrados, rango [0,2]. El
Brier binario [0,1] de compatibilidad es una métrica diferente.

La identidad del modelo incluye activo, horizonte, parámetros, revisión del
código y versión del evento. El contrato y las bandas congeladas están dentro
del JSON firmado de la observación. Las consultas verifican ambos hashes,
excluyen resultados futuros respecto al corte y no reinterpretan registros
binarios antiguos como evidencia multiclase. No se requiere una nueva migración.

## Interpretación y limitaciones

Se exige al menos 500 observaciones efectivas, 100 de calibración, 100 de
holdout, las tres clases en calibración y variación de scores. Sin esa evidencia
se conserva la distribución original como **Score heurístico preliminar**.
Cumplir estos mínimos permite evaluar una curva, pero NO garantiza buen Brier,
rentabilidad ni seguridad para operar; compare Brier con controles y revise
las reliability curves. El score operativo global sigue siendo heurístico:
no toma prestado el calibrador de otro horizonte ni cambia vetos u órdenes.

La evaluación es una ventana móvil retrospectiva, no un holdout permanente.
Consultar repetidamente su desempeño para escoger modelos exige un futuro
test sellado adicional. La purga elimina solapamientos de etiquetas, pero no
prueba independencia estadística completa entre sesiones correlacionadas.

## Integridad y verificación

No se modifican caja USD/MXN, operaciones, respaldos ni cifrado. Las pruebas
usan bases temporales. Se añade `tests/test_calibration_oos.py`; del conjunto
anterior solo se actualiza, con autorización, el caso que exigía Brier in-sample.
