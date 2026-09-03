# Alcance de zonas durante la sesión actual

## Modelo activo: conditional-v3-weighted

El adaptador usa `matching='weighted'`: combina 25% de peso uniforme entre las
sesiones completas del mismo horario y 75% de peso por semejanza. No se vuelve
al score direccional ni se exige que todos los indicadores coincidan exactamente.
La distancia simétrica compara siete aspectos: signo semanal, signo diario,
régimen ADX, posición EMA50, ATR porcentual, consumo del rango y movimiento desde
la apertura normalizado por ATR. El tiempo restante se mantiene igual en todas
las observaciones. La ponderación no emplea el resultado futuro de cada sesión.

Pesos y descuento ATR son supuestos de modelado NO optimizados ni calibrados OOS.
Se requieren 12 sesiones completas y 8 observaciones efectivas; se permite una
estimación PRELIMINAR de evidencia limitada, no la antigua afirmación de muestra
condicionada suficiente. n efectivo = 1/suma(pesos normalizados al cuadrado).
El Wilson aproximado usa ese n efectivo, no el número bruto. No es un certificado
de cobertura estadística en presencia de dependencia temporal o cambio de régimen.

Las seis zonas conservan probabilidades separadas de toque y cierre. Falta de
histórico macro, mercado cerrado, cotización atrasada o datos incompletos todavía
producen N/D: no hay porcentajes inventados ni uso de otra sesión como si fuera hoy.
La pantalla y los cuatro PDF usan la misma captura de resultados.

## Modelo anterior: conditional-v2 (modo estricto conservado)

La pantalla y los cuatro PDF usan `conditional_zone_reach.py`. El modelo
v1 descrito abajo permanece disponible para compatibilidad, pero NO es un
fallback del modelo activo.

Cada sesión candidata debe coincidir exactamente en signo EMA9/21 semanal,
signo EMA9/21 diario, régimen ADX diario (<20, 20-25, >25) y posición del cierre
respecto a EMA50 diaria. Se usan las últimas velas completas anteriores a la
sesión, tanto para hoy como para cada candidato; no el cierre futuro de ese día.
Las semanas incompletas no participan. No es un filtro anual o fundamental.

Se exige cobertura OHLC completa de sesiones de igual duración y al menos 20
sesiones coincidentes. Menos evidencia devuelve N/D, sin relajar filtros.
El histórico intradía existente de un mes puede ser insuficiente tras filtrar.

Las excursiones logarítmicas restantes se escalan por la razón entre el presupuesto
actual e histórico: `ATR/precio * sqrt(1-t) / (1+c*max(1,c/sqrt(t)))`, donde `t`
es la fracción de sesión transcurrida y `c` el rango high-low consumido / ATR14
diario previo. Es un presupuesto suave, NO un límite al rango posible.
La comparación a la misma hora incorpora el tiempo restante empíricamente;
no se aplica un descuento adicional arbitrario al porcentaje. La probabilidad
no tiene que decrecer si el precio se acerca al objetivo o cambia la volatilidad.

Toque: cruzar la frontera cercana o estar dentro de la zona al corte. Cierre:
cierre final <= frontera inferior de compra, o >= frontera superior de venta.
Se informan ambos eventos, sus intervalos Wilson y el número de sesiones.
Cerrar debajo de un soporte no implica un rebote ni una compra recomendable.

Son estimaciones condicionadas y ajustadas, NO precisión validada: la fórmula
de presupuesto requiere evaluación walk-forward y calibración OOS por evento.
Los intervalos solo reflejan variabilidad muestral, no error de modelo ni
dependencia entre sesiones. No se modifica ningún umbral operativo ni posición.

## Modelo v1 (compatibilidad)

`analytics/zone_reach.py` calcula una frecuencia histórica de primer alcance,
independiente de los scores del motor y de sus decisiones de ejecución.

- Referencia: último cierre de 5 minutos disponible, no cotización tick a tick.
- Horizonte: desde ese corte hasta el cierre de la sesión XNYS actual, incluidos
  horario de verano y cierres anticipados. Fuera de sesión no predice mañana.
- Observación: una sesión anterior con cobertura completa del mismo tramo horario.
  Se dividen sus extremos posteriores por el cierre de referencia histórico y
  se trasladan proporcionalmente al precio actual. Se cuenta si alcanzan la
  frontera más cercana de cada zona. No se incluyen velas actuales en formación,
  días futuros ni retornos nocturnos.
- Estimación: sesiones que alcanzaron el nivel / sesiones válidas. Mínimo 20;
  cada sesión cuenta una sola vez. Intervalo Wilson del 95% y tamaño de muestra
  acompañan cada porcentaje. Una zona que contiene el precio se identifica como
  alcanzada al corte, no como una predicción de éxito.
- Una cotización con más de diez minutos desde su cierre se considera atrasada.
  Niveles ausentes, histórico incompleto o muestra insuficiente producen N/D.

## Límites

Es una probabilidad **estimada por frecuencia histórica**, no una probabilidad
real conocida ni calibrada fuera de muestra. No condiciona por noticias, régimen
actual o volatilidad actual. Supone comparabilidad de los retornos porcentuales
del tramo horario entre sesiones. La dependencia entre días puede estrechar
artificialmente el intervalo Wilson; éste no cubre el error de modelo.
Los extremos OHLC indican cruce/alcance aproximado, no garantizan una ejecución
en el nivel si hubo un salto. No estima rentabilidad, ni alcanzar TP antes de SL.

La descarga existente de un mes puede aportar menos de 20 sesiones completas;
en tal caso se muestra N/D en lugar de fabricar evidencia o multiplicar muestras
con simulaciones. No se modificaron las descargas ni los indicadores existentes.

El adaptador visual vincula los resultados por orden a las seis zonas. No escribe
en SQLite, no cambia posiciones/órdenes, ni usa o modifica credenciales,
respaldos, contabilidad o estados de ejecución.
