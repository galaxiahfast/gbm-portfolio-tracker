# Portafolio GBM+

Aplicación local para controlar efectivo MXN/USD, operaciones de acciones de Estados Unidos/SIC, comprobantes de GBM+, posiciones FIFO, rendimiento y auditoría. Al iniciar por primera vez registra automáticamente el capital inicial solicitado de **921.05 USD**, una sola vez.

## Inicio rápido en Windows

Requiere **Python 3.11 o posterior**.

1. Haz doble clic en `iniciar_app.bat`.
2. La primera ejecución crea un entorno privado e instala `requirements.txt`.
3. Se abrirá la aplicación en el navegador. Los datos quedan en `data/portfolio.db` y las imágenes en `data/receipts/`.

Alternativa desde PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --use-feature=truststore --upgrade pip setuptools wheel
python -m pip install --use-feature=truststore --no-build-isolation -r requirements.txt
streamlit run app.py
```

## Autollenado OCR

El autollenado funciona de fábrica con **RapidOCR + ONNX**, cargados en memoria únicamente cuando analizas una imagen. Si ya tienes **Tesseract OCR 5**, la aplicación lo prefiere automáticamente. En Windows Tesseract suele quedar en:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

Si Tesseract está en otra ubicación, define `TESSERACT_CMD` antes de iniciar:

```powershell
$env:TESSERACT_CMD = "D:\Apps\Tesseract-OCR\tesseract.exe"
streamlit run app.py
```

El flujo es deliberadamente asistido: OCR → campos editables → doble conciliación aritmética → confirmación humana → guardado. El sistema no ejecuta órdenes bursátiles ni guarda una lectura OCR sin aprobación.

El formulario distingue entre:

- **Total bruto mostrado:** títulos × precio; la comisión aparece separada, como en el comprobante GBM adjunto.
- **Cargo o abono final:** compra = títulos × precio + comisión; venta = títulos × precio − comisión.

La tolerancia del total se limita al error matemáticamente posible por el precio unitario mostrado a centavos, más un centavo de cierre.

## Qué registra

- Ingresos originalmente en MXN y su conversión a USD con la tasa aplicada.
- Retiros solicitados en USD y la salida equivalente en MXN.
- Compras y ventas, títulos, precio, total ejecutado, comisión, fecha, producto y tipo de orden.
- Comprobante original exacto, huella SHA-256 y miniatura WEBP; SQLite solo conserva rutas y metadatos para no inflar la memoria ni la base.
- Posiciones por lotes FIFO, costo promedio, P&L realizado/no realizado, comisiones y rendimiento sobre aportaciones netas.
- Cotizaciones y valuaciones históricas para la gráfica de evolución.
- Auditorías reproducibles de SQLite, efectivo, títulos, conciliación y SHA-256.
- Predictor Fase 4 multi-temporal con confluencia estricta y veto de riesgo.

## Fuentes de mercado

- La cotización intradía consulta un endpoint ligero de Yahoo Finance, hasta cuatro emisoras en paralelo.
- Si USD/MXN no responde, se usa Frankfurter como respaldo de referencia publicado por bancos centrales.
- Las respuestas se conservan cinco minutos y se refrescan en segundo plano. Si la red falla se usa el último valor local.
- Siempre se muestra fuente y fecha. En `Configuración` puedes guardar la tasa o precio exacto aplicado por GBM.

Una tasa pública es indicativa: puede diferir por horario, spread y liquidación. Para la contabilidad manda el comprobante de GBM.

## Estructura

```text
app.py                              Interfaz Streamlit
portfolio_tracker/config.py         Rutas, zona horaria y capital inicial
portfolio_tracker/db.py             Esquema SQLite y tablas futuras de ML
portfolio_tracker/repository.py     Persistencia y consultas
portfolio_tracker/models.py         Modelos y precisión Decimal
portfolio_tracker/services/audit.py Integridad contable, SQLite y SHA-256
portfolio_tracker/services/quant_market_data.py Descarga y normalización yfinance
portfolio_tracker/analytics/technical_probability.py Indicadores y motor heurístico
portfolio_tracker/analytics/multi_timeframe.py Contexto macro, liquidez y veto central
portfolio_tracker/services/ocr.py   OCR y parser adaptable GBM
portfolio_tracker/services/validation.py  Conciliación doble
portfolio_tracker/services/market_data.py Proveedores reemplazables
portfolio_tracker/services/portfolio.py   FIFO, efectivo y P&L
portfolio_tracker/analytics/        Contratos para riesgo y predicción
tests/                              Pruebas contables y de OCR
```

## Preparación para análisis futuro

`AnalyticsModule` y `PredictionModel` evitan acoplar la aplicación a una librería concreta. SQLite ya incluye `analytics_features`, `prediction_runs` y `predictions`, con versión, horizonte y fecha de corte. Un futuro modelo deberá:

- separar análisis fundamental, técnico y gestión de riesgo;
- registrar supuestos, procedencia de datos y versión;
- entrenar sin usar información futura y validar fuera de muestra;
- calibrar probabilidades y mostrar incertidumbre;
- aplicar margen de seguridad, diversificación y límites de riesgo;
- no convertir una probabilidad en una recomendación automática.

Estos criterios recogen conceptualmente disciplina, riesgo/recompensa y lectura técnica del material de trading adjunto, junto con margen de seguridad, diversificación y separación entre inversión y especulación de Graham.

## Predictor de probabilidades · Fase 4

La página analiza SMCI por defecto y acepta otro ticker de Estados Unidos. Realiza solo dos descargas por emisora: un mes de velas a 5 minutos y cinco años diarios, con caché de cinco minutos, actualización en segundo plano y máximo de 12 emisoras almacenadas. Las vistas de 1 hora, semanal y mensual se derivan localmente. Calcula:

- Stoch RSI (14,14,3,3) y Bandas de Bollinger (20,2);
- cruce %K/%D en sobrecompra o sobreventa y contacto con banda;
- volumen actual frente a la media de las 20 velas anteriores;
- pivote diario, S1, S2, R1 y R2;
- EMA 9, 21, 50 y 200 en contexto diario y semanal;
- MACD (12,26,9) intradía y diario, con veto si contradice una señal confirmada;
- VWAP reiniciado por sesión y posición relativa del precio;
- ADX 14 con reducción explícita de fiabilidad por debajo de 20;
- OBV y divergencia precio/volumen contra las 12 velas anteriores;
- Fibonacci 0.382, 0.500 y 0.618 sobre las 22 sesiones completas anteriores, con tolerancia de zona del 0.35%;
- Ichimoku 9/26/52 en 5 minutos y diario, incluida la posición frente a la nube y la relación Tenkan/Kijun;
- martillo, estrella fugaz, envolventes y velas básicas de continuación sobre la última vela cerrada;
- Fibonacci anual sobre 252 sesiones y tres zonas aproximadas de liquidez por volumen;
- tendencias semanales y mensuales firmes mediante precio, EMA21, EMA50 y MACD;
- veto central que limita la probabilidad de la operación a 39% cuando el macro contradice el gatillo, ADX es menor a 20 o existe sobreextensión diaria sin volumen;
- puntaje ponderado auditable de subida/bajada y nivel técnico de vigilancia.

La interfaz superior separa dos experiencias sin repetir las descargas:

- **Vista Ejecutiva (Modo Rápido):** decisión concreta, zona de entrada, stop loss, dos take profits y distribución subida/rango/bajada para 1 hora, 6 horas, 1 día, 1 semana, 1 mes y 6 meses. La tabla conserva esos horizontes como mapa comparativo. La gráfica principal es independiente: proyecta estrictamente Día 1 a Día 15 sobre sesiones hábiles, con cierre esperado y piso/techo derivados de retorno reciente, contexto EMA/MACD, confluencia y ATR diario.
- **Vista Técnica Avanzada (Completa):** conserva los paneles intradiario, diario/semanal y mensual/anual, todas las gráficas técnicas y la tabla auditable de aportes del motor.

Las pestañas son dinámicas: la vista técnica solo construye sus gráficas cuando se abre. El PDF se genera en memoria con ReportLab, no escribe datos del usuario en el repositorio e incluye la trayectoria vectorial de 15 sesiones, Bandas de Bollinger/VWAP, Estocástico RSI, MACD intradía y estructura EMA diaria construidos desde los mismos DataFrames reales de la vista avanzada, además del bloque estructurado para revisión por otra IA.

La barra superior del predictor ofrece tres descargas consistentes con el mismo corte de mercado: Vista Ejecutiva, Vista Técnica Avanzada y Reporte Completo. El PDF técnico incorpora quince paneles disponibles: Bollinger/VWAP, Estocástico RSI, MACD 5m y 1h, ADX/+DI/-DI, OBV, estructuras EMA diaria/semanal/mensual, MACD diario/semanal/mensual, Ichimoku diario y dos paneles de patrones chartistas (5m y diario). Si una temporalidad todavía no reúne observaciones suficientes, el reporte lo indica sin cancelar las demás páginas.

El detector de patrones usa pivotes Zig-Zag dependientes del ATR y evalúa dobles/triples techos o suelos, rupturas de rango con volumen superior a 1.2x y microestructuras de tres impulsos/ABC. Solo una figura confirmada con más de 75% modifica el motor; una figura bajista confirmada en 5m veta una compra y una figura alcista equivalente veta una venta. Las formaciones incompletas se muestran para auditoría, pero no alteran probabilidades.

La navegación incluye **Control de implementación**, un inventario local que recalcula módulos, líneas útiles y pruebas descubiertas, y diferencia esas métricas de la última suite completa aprobada.

Los objetivos monetarios de la tabla combinan ATR de Wilder de 14 periodos escalado por horizonte, Bandas de Bollinger de 20 periodos y soportes/resistencias locales. La probabilidad es simétrica: el MACD de 5 minutos, su aceleración, el retorno de las seis velas cerradas más recientes y la distancia al VWAP dominan el sesgo inmediato; el veto reduce la conveniencia de operar sin invertir la lectura direccional. La proyección diaria usa bloques bootstrap de retornos históricos, una deriva limitada por ATR, reacción suave a soportes/resistencias y percentiles 15–85 de 320 trayectorias reproducibles. Las fechas son días hábiles de lunes a viernes; no sustituyen un calendario oficial de feriados bursátiles. Son escenarios informativos, no velas observadas, órdenes ni precios garantizados.

El puntaje parte de 50% y cada filtro aporta puntos visibles en la interfaz. Sin señal principal se limita a 40–60%; una señal solo vigilada se limita a 35–65%. Una señal confirmada que contradiga el MACD de 5 minutos queda rechazada. La regla de oro de Fase 4 tiene prioridad sobre todos los bonos y produce una recomendación concreta de evitar la operación cuando se activa. Este resultado sigue siendo heurístico, no una probabilidad calibrada con resultados históricos. El módulo no genera ni ejecuta órdenes.

## Verificación

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## Migraciones y persistencia segura

En cada arranque la aplicación:

1. verifica y crea cualquier tabla base faltante con sentencias idempotentes;
2. consulta `schema_migrations`;
3. crea una copia SQLite consistente en `data/backups/` si la base ya existía;
4. aplica solamente las migraciones pendientes;
5. conserva intactos los registros existentes.

La base `data/portfolio.db`, sus archivos WAL/SHM y `data/receipts/` están ignorados explícitamente en `.gitignore`. Por ello un `git pull` o `git push` normal no los reemplaza ni los publica. Si una base fue agregada a Git antes de usar este `.gitignore`, hay que retirarla del índice una sola vez, sin borrarla del disco:

```powershell
git rm --cached data/portfolio.db
git rm -r --cached data/receipts
git commit -m "Dejar datos privados fuera de Git"
```

No uses `git clean -fdx` en este proyecto: ese comando sí puede borrar archivos ignorados.

## Auditoría y respaldo

La sección **Auditoría** comprueba:

- integridad interna de SQLite y versión del esquema;
- saldo reconstruido desde ingresos, retiros y operaciones;
- deltas de efectivo de cada compra/venta;
- ausencia de títulos negativos en toda la secuencia histórica;
- conciliación de totales y comisiones;
- existencia y SHA-256 de cada comprobante;
- imágenes huérfanas, vínculos duplicados y operaciones sin imagen.

Cierra la aplicación y copia la carpeta `data/`. La base y los comprobantes deben respaldarse juntos. No compartas esa carpeta si contiene información personal.
=======
# gbm-portfolio-tracker
Aplicación web interactiva en Python para el control de portafolio en GBM+ (SIC/EUA) con gestión de divisas MXN/USD, lectura de comprobantes por OCR, contabilidad FIFO y arquitectura modular extensible para análisis predictivo.
