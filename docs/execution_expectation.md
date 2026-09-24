# Expectativa operativa con gaps y fills

La columna **teórica** supone entrada en el precio de referencia, salida en el
stop u objetivo exacto y los costes configurados. Se conserva únicamente como
comparación; no autoriza una orden.

La columna **observada/realista** usa las salidas `SL_FIRST` elegibles del
conjunto de *desarrollo* del mismo horizonte y únicamente para entradas LONG.
Cada pérdida se expresa como `(entrada - salida_real) / (entrada - stop)`.
Una apertura por debajo del stop produce un múltiplo mayor que 1. El holdout
final no aporta pérdidas a esta distribución, para evitar reutilizarlo en la
decisión. Si no existen suficientes salidas observadas, el modelo no se
aprueba; si faltan en una evaluación individual, la oportunidad no es apta.

Los fills todavía no son observaciones reales de GBM+. Mientras no exista un
histórico verificable de órdenes enviadas y ejecutadas, se aplican **supuestos
explícitos**: 5% de no ejecución, spread completo de 4 puntos básicos (mitad
al comprar y mitad al vender) y 5 puntos básicos de deslizamiento por lado,
además de la comisión configurada. Una operación no ejecutada aporta cero al
valor esperado; el timeout se estresa a la peor pérdida SL observada.

La posición se dimensiona con esa peor pérdida observada, efectivo disponible
y concentración máxima. En `BOTH_REDUCED`, el número propuesto de acciones se
multiplica por el factor de exposición (máximo 25%) **antes** de emitir
`COMPRAR`; si el redondeo deja cero acciones, se recomienda esperar.

Estos datos mejoran la comparación, pero no garantizan un límite de pérdida:
un futuro gap puede ser peor que cualquiera del desarrollo. Antes de llamar
"empírico" al modelo de fills se necesitarán registros de órdenes no
ejecutadas, cotizaciones bid/ask y precio de ejecución punto-en-tiempo.
