# Selección de operaciones por expectativa neta

La decisión de una posición LONG nueva ya no escoge el horizonte con mayor
`score_alcista - score_bajista`. Primero exige un artefacto firmado del
walk-forward anidado V4 para el símbolo y el horizonte, aprobado con Brier y
log-loss inferiores al score crudo y al baseline tanto en OOF posterior como
en el holdout final. No usa modelos de otro activo ni un artefacto futuro con
respecto a la vela cerrada del corte. Si la última corrida válida fue
rechazada, tampoco recupera silenciosamente un modelo aprobado anterior.

Para cada horizonte aprobado, la entrada, el stop y el TP proceden del mismo
plan LONG de la vela cerrada. El vector de features se construye con el mismo
contrato del replay histórico. Las tres probabilidades calibradas describen
`TP_FIRST`, `SL_FIRST` y `TIMEOUT`; no se sustituyen por el score direccional
UP/RANGE/DOWN ni por la frecuencia de toque de una zona.

## Cálculo por acción

Por defecto se descuentan 25 puntos base de comisión y 5 de deslizamiento
**en cada lado** (configurables en `TradingCostPolicy`). La columna teórica
conserva la salida al stop exacto únicamente como comparación:

```text
beneficio_TP_neto = TP - entrada - coste_entrada - coste_salida_TP
perdida_SL_neta = stop - entrada - coste_entrada - coste_salida_stop
EV_neta = P(TP_FIRST) * beneficio_TP_neto
          + [P(SL_FIRST) + P(TIMEOUT)] * perdida_SL_neta
R:R_neto = beneficio_TP_neto / abs(perdida_SL_neta)
```

La columna **observada/realista**, usada para seleccionar la operación, toma
la distribución de salidas `SL_FIRST` LONG del desarrollo histórico (sin el
holdout final). Incorpora spread y no-fill supuestos; el timeout se estresa a
la peor salida SL observada. Consulta [execution_expectation.md](execution_expectation.md)
para las fórmulas, supuestos y límites. Sin pérdidas observadas suficientes no
se autoriza la entrada.

El tamaño entero de posición es el mínimo de tres límites: peor pérdida SL
observada <= 2% del patrimonio, efectivo suficiente incluyendo coste de entrada,
y valor de posición <= 30% del patrimonio. Entre oportunidades con EV neta
positiva y R:R neto >= 1.5, se elige la de **mayor EV neta total en USD**, no
el mayor score o porcentaje de acierto. El veto de riesgo, permiso macro y
gatillo de vela cerrada siguen siendo obligatorios. `BOTH_REDUCED` aplica
su factor de exposición al número de acciones antes de emitir `COMPRAR`.

Ninguna distribución histórica acota un gap futuro. La cifra es informativa;
nunca envía órdenes a GBM+.

## Estado de evidencia

La aprobación depende del número de entradas ejecutables, las clases presentes
en validación y holdout, y pérdidas SL LONG observadas en desarrollo. Una
ausencia de cualquiera de estos insumos mantiene la señal en ESPERAR.
