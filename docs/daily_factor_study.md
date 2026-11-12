# Estudio diario de factores documentados (V1, 11-nov-2026)

**Estado: INVESTIGACIÓN.** No modifica `portfolio.db`, el gatillo ni la app; no
autoriza compras o ventas.

## Qué se probó y cómo

- **Datos:** velas diarias ajustadas (splits y dividendos) de Yahoo, 2004-01 a
  2026-11, para 97 acciones grandes de EE. UU. más SPY. El efectivo rinde la
  tasa del T-bill a 13 semanas (`^IRX`). Los precios quedan solo en
  `output/daily_study/data/` (ignorado por Git) con huella SHA-256.
- **Parámetros fijados antes de ver resultados** (valores de libro, sin
  optimizar): momentum 12-1 con el quintil superior y rebalanceo mensual;
  tendencia con SMA 200 revisada a fin de mes; objetivo de volatilidad de 15%
  anual con EWMA λ = 0.94, sin apalancamiento y revisado cada 5 días.
- **Causalidad:** se decide con el cierre de *t*, se ejecuta al cierre de
  *t+1* y se gana desde *t+2*. Entre operaciones los pesos derivan con el
  precio; no hay rebalanceo implícito.
- **Costos:** 30 pb por lado (0.25% de comisión GBM + 5 pb de deslizamiento)
  sobre cada unidad de rotación.
- **Periodos:** desarrollo 2005–2018 y reserva 2019–2026. Como no hubo ajuste
  de parámetros, ambos tramos son evaluación; la reserva mide estabilidad.
- **Incertidumbre:** intervalos al 95% con *bootstrap* estacionario por
  bloques de 21 días (1,000 remuestreos), pareado contra su referencia.
- **Calidad de datos:** se revisaron todos los movimientos diarios mayores a
  50%. Cuatro son reales (C y MS en 2008, AMD en 2016, OXY en 2020). El de DHR
  del 2016-07-05 (+61%) es la escisión de Fortive mal ajustada por el
  proveedor; se neutralizó de forma explícita y queda registrada en el reporte.

## Sesgo conocido

El universo son las grandes empresas *de hoy*: faltan las que quebraron o
salieron del índice. Por eso "igual peso del universo" supera a SPY por
~4.4%/año, y casi todo eso es sesgo de supervivencia, no habilidad. El
momentum se compara contra ese mismo universo, que carga el mismo sesgo.

## Resultados (completo 2005–2026, después de costos)

| Estrategia | CAGR | Volatilidad | Sharpe | Caída máx. | Costo/año |
|---|---:|---:|---:|---:|---:|
| Comprar y mantener SPY | 10.9% | 18.9% | 0.55 | −55.2% | 0.00% |
| Tendencia SMA 200 | 8.5% | 12.5% | 0.58 | −27.4% | 0.41% |
| Objetivo de volatilidad 15% | 9.6% | 13.3% | 0.62 | −33.5% | 0.35% |
| Igual peso del universo | 15.9% | 18.8% | 0.78 | −47.7% | 0.19% |
| Momentum 12-1 | 17.5% | 22.7% | 0.75 | −52.3% | 1.86% |
| Momentum 12-1 + tendencia | 15.5% | 17.8% | 0.80 | −26.5% | 1.83% |

Diferencias contra su referencia (IC 95%):

- **Tendencia vs SPY:** −3.2%/año [−7.9%, +1.9%]; Sharpe +0.03 [−0.29, +0.35].
  En la reserva 2019–2026 rindió −8.1%/año menos, por las salidas y
  reentradas tardías de 2020 y 2022.
- **Objetivo de volatilidad vs SPY:** −2.1%/año [−4.7%, +0.7%]; Sharpe +0.07
  [−0.09, +0.23]. En la reserva bajó la caída máxima de −33.7% a −19.8% con
  un Sharpe igual (0.75 vs 0.77).
- **Momentum vs igual peso:** +2.2%/año [−2.5%, +6.9%], con Sharpe igual o
  menor. La rotación de GBM cuesta ~1.9%/año y se come casi toda la ventaja.
- **Momentum + tendencia vs igual peso:** −0.5%/año [−6.8%, +6.2%]. Funcionó
  en desarrollo (Sharpe 0.98) y falló en la reserva (0.54): eso es inestable,
  no una ventaja.

**Ninguna regla supera a su referencia con significancia.** Los filtros de
tendencia y de volatilidad cambian rendimiento por menos caídas; no lo
aumentan.

## Pronóstico de volatilidad (lo que sí funciona)

Para la varianza realizada de los próximos 5 días, con GARCH(1,1) reajustado
cada año usando solo el pasado (QLIKE: menor es mejor):

| Activo | Ingenuo 20 días | EWMA 0.94 | GARCH walk-forward | Mejora GARCH | Años en que GARCH gana |
|---|---:|---:|---:|---:|---:|
| SPY | 0.588 | 0.515 | 0.462 | 21% | 17 de 19 |
| SMCI | 1.149 | 0.957 | 0.850 | 26% | 13 de 16 |
| NVDA | 0.659 | 0.586 | 0.552 | 16% | 17 de 19 |

La mejora se mantiene en la reserva 2019–2026 para los tres activos. La
dirección del precio casi no se puede predecir; **el tamaño del movimiento,
sí**. Su uso correcto es dimensionar posiciones, no elegir dirección.

Volatilidad diaria pronosticada al 11-nov-2026: SPY 1.6%, NVDA 3.6%, SMCI
5.0% (≈79% anual). Con la regla "pérdida de 1 día al 99% ≤ 1% del capital",
el tope por cada $1,000 de capital es ≈$264 en SPY, $120 en NVDA y $86 en SMCI.
Las colas reales son más gruesas que la normal, así que esto es un techo, no
una garantía.

## Reproducir

```powershell
.venv\Scripts\python.exe scripts\run_daily_factor_study.py            # descarga (caché diaria) y ejecuta
.venv\Scripts\python.exe scripts\run_daily_factor_study.py --offline  # reusa la caché más reciente
```

El reporte firmado queda en `output/daily_study/report_<UTC>.json`.
