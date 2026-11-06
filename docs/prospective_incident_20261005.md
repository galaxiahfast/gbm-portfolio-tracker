# Incidente de cortes 10:00 NY — 01 y 05 oct 2026

## Evidencia y causa

El registro `System` de Windows, proveedor `Microsoft-Windows-Power-Troubleshooter`, evento 1, muestra:

| Corte esperado | Suspensión anterior (UTC) | Despertar (UTC) | Retraso frente al corte |
|---|---|---|---:|
| 01-10 14:00 UTC (10:00 NY) | 01-10 00:08:36 | 01-10 14:46:27 | 46 min 27 s |
| 05-10 14:00 UTC (10:00 NY) | 03-10 19:15:44 | 05-10 14:51:41 | 51 min 41 s |

El equipo estuvo suspendido en ambos instantes. El colector solo acepta cinco minutos después de cada corte y por diseño no firma predicciones retrospectivas. Los snapshots de las 11:00 de esos días sí aparecieron tras el despertar. No hay llamadas ni errores de API correspondientes a los cortes ausentes en `collector.log`. El historial detallado `Microsoft-Windows-TaskScheduler/Operational` estaba `enabled: false`, así que no permite atribuir un código de salida de tarea en esos momentos. La causa demostrada es **suspensión del host durante el corte**; una caída de API/timeout no explica la ausencia de invocación.

## Mitigación

`scripts/register_prospective_tasks.ps1` registra una cuarta tarea, `STOCK-Prospective-Cut-Wake`, con `WakeToRun=true`. Marca un despertar tres minutos antes de los horarios de corte, con ambas variantes de horario local NY/México. El colector existente sigue ejecutándose cada minuto y el guard XNYS es el único que autoriza crear snapshots. La tarea de despertar no descarga datos ni usa claves. Se verificó que Windows programa el siguiente marcador para el 05-10 a las 10:27 hora local; en corriente alterna los temporizadores de reactivación están habilitados. En batería están deshabilitados. La medida tampoco puede encender una PC completamente apagada ni garantizar conectividad/API; en esos casos el reporte registra el faltante y no lo reemplaza con un falso PIT.

Verificación posterior: el marcador se ejecutó el 05-10 a las 16:27:25 UTC y el colector guardó automáticamente los cuatro snapshots del corte 12:30 NY entre 16:31:45 y 16:31:47 UTC. Un intento del programador devolvió transitoriamente `0x800710E0` durante un solapamiento de tarea con `IgnoreNew`, pero el siguiente intento terminó con resultado `0` y no faltó ningún registro de ese corte. La ejecución manual posterior devolvió `ALREADY_COLLECTED` sin crear duplicados.

## Reconstrucción de investigación, no backfill PIT

Se guardaron ocho archivos firmados (dos fechas × dos activos × dos horizontes) en `output/prospective_archive/backfills/`. Su `cut_at_utc` es el instante **real** de reconstrucción del 05-10, no el corte perdido; `historical_cut_at_utc` identifica el corte retrospectivo. Todos tienen `backfilled=true`, `pit_quality=degraded`, `training_eligible=false`, `production_actionable=false`, sin sentimiento ni resultado operativo inventado. Las barras IEX se consultaron después y podrían haber sido revisadas. La cobertura prospectiva mantiene los dos cortes como ausentes y muestra los backfills solo por separado.

No se tocó `portfolio.db`, no se entrenó ningún modelo, no se cambió el gatillo ni se promovió una señal.
