# Plataforma de Scoring Alternativo para PYMEs (Flask)

Prototipo funcional desarrollado para **Reto 2 — Evaluación Inteligente de Riesgo Financiero para PYMEs (Viamática)**.

El sistema utiliza **datos financieros** y **datos no financieros** (información básica de redes sociales) para clasificar a las empresas en **riesgo bajo, medio o alto**, mostrando el puntaje total y las variables clave que influyen en la evaluación.

---

## Inicio rápido (local)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
export FLASK_APP=app.py  # En Windows: set FLASK_APP=app.py
flask run --port 8501

```
# Abrir en navegador:  http://127.0.0.1:8501  

---

## Estructura
```
reto_2/
├─ app.py                     # Rutas Flask y flujo principal
├─ funtions.py                # Funciones para procesar y evaluar empresas
├─ nlp.py                     # Análisis de texto y funciones auxiliares de lenguaje
├─ requirements.txt           # Dependencias del proyecto
├─ data/                      # Archivos CSV y de prueba
│   ├─ bi_ranking.csv         # Data obtenida de la base de datos desde el 2008 con mas de 54 variables financieras
│   ├─ df_financiero.csv      # Data con variables financiera
│   ├─ df_pruebas.csv         # Data no entrenada para pruebas de predicción
│   ├─ df_score.csv           # Data de entrenamiento con score y calificación
│   └─ predicciones.csv       # Dataobtenida de aplicación de modelo SVM
├─ docs/                      # Documentación técnica y pitch
│   ├─ technical_report.md
│   └─ pitch_outline.md
├─ models/                    # Modelos entrenados y objetos de preprocesamiento
│   ├─ modelo_svm.joblib      # Modelo SVM obtenido para su llamada
│   ├─ scaler.joblib          # Indice de escalabilidad de datos para realizar predicciones
│   └─ feature_columns.json   # Columnas que intervienen en el entrenamiento
├─ static/css/styles.css      # Estilos personalizados
├─ templates/                 # Plantillas HTML
│   ├─ base.html
│   ├─ index.html             # Formulario de carga y consulta
│   └─ result.html            # Vista de resultados con gráficos
├─ tests/test_scoring.py      # Pruebas unitarias
└─ utils/                     # Scripts de apoyo para el modelo
    ├─ data_financiera.py     # Filtrado y clasificación de la data general
    ├─ data_ml.py             # Preparación de data para entrenamiento
    ├─ ML_clasification.py    # Implementación de modelo SVM y almacenamiento de Modelo 
    ├─ predict.py             # Implementación de la predicción de SVM para ver su confiabilidad
    └─ scraper.py             # Implementación de scraping para obtener indormación de redes sociales
```
---

## Entradas esperadas

- **Certificado PDF** de la Superintendencia de Compañías (obligatorio).
  - El sistema extrae automáticamente:
    - **Razón Social**
    - **Expediente**
    - **RUC**
- **URL de red social** (opcional): Facebook o Instagram del negocio.
  - Se usa para recolectar datos relevantes de la red social (seguidores y publicaciones) para generar una gráfica **variables recolectadas** y poder determinar su presencia en redes sociales.

> Si el PDF no contiene un **Expediente** válido, se muestra un mensaje de error y no se muestran resultados.

---

## Flujo de procesamiento

1. **Carga del PDF** y, opcionalmente, **URL** desde `index.html`.
2. **Extracción de texto del PDF** (pdfplumber) → parseo robusto de:
   - Razón Social, Expediente y RUC.
3. **Validación por Expediente**:
   - Busca el expediente en `df_score.csv`.
     - Si **existe**: se construye el vector de entrada, se escala y se predice con el modelo SVM.
     - Si **no existe**: revisa `bi_ranking.csv` (data general).
       - Si está en **bi_ranking** → **“Su compañía no es PYME.”**
       - Si no aparece tampoco → **“Alerta: no existe expediente en la Super de Compañías.”**
4. **(Opcional) Reputación**:
   - Si se proporcionó URL, se obtienen reseñas simples (scraper básico) y se extrae la sigueinte información.
      - Dirección completa de red social
      - Tipo de plataforma
      - Número de seguidores
      - Número de publicaciones
      - Opcional Número de seguidos (solo Instagram)
   - Este valor **no altera** el modelo SVM entrenado; se muestra como dato complementario si está disponible.
5. **Visualización de resultados**:
   - Muestra datos del certificado, **Score** y **Riesgo** (semáforo), y una **gráfica de barras** con los valores normalizados (0–1) de las **mismas features** usadas por el modelo y a su vez la parte de analisis de redes sociales.

---

## Resultados mostrados

- **Datos de la empresa**: Razón Social, RUC, Expediente (extraídos del PDF).
- **Score** (si aplica): valor en escala 0–1.
- **Riesgo** (clasificación del SVM): **Bajo / Medio / Alto** (con semáforo de color).
- **Variables del modelo (chart)**: gráfico de barras **Chart.js** con las features normalizadas (0–1) en orden.
- **Mensajes de estado**:
  - **No PYME** (si el expediente aparece en la data general).
  - **Expediente inexistente** (si no aparece en ninguno).
  - **Error** (si el PDF no trae los campos requeridos).

---

## Modelo y datos

- **Entrenamiento**:
  - Fuente: `data/df_score.csv` (variables ya normalizadas + `score_final` + `riesgo`).
  - Target: `riesgo` (0=Bajo, 1=Medio, 2=Alto).
  - Modelo: `SVC(kernel="rbf", C=1.0, gamma="scale", probability=False)`.
  - Escalado: `StandardScaler` (guardado en `models/scaler.joblib`).
  - Columnas de entrada (orden exacto): `models/feature_columns.json`.
- **Inferencia**:
  - Siempre se construye **X** usando **esas mismas columnas** y orden.
  - Se aplica **el mismo scaler** y luego `predict` del SVM.

---

## Archivos de datos relevantes

- `data/df_score.csv` — **Fuente principal** para inferencia (matching por **expediente**).
- `data/bi_ranking.csv` — Catálogo para verificar si la empresa **no es PYME**.
- `models/modelo_svm.joblib` — Modelo entrenado.
- `models/scaler.joblib` — Escalador usado en el entrenamiento.
- `models/feature_columns.json` — Columnas/orden exactos del modelo.

---

## Interfaz

- `index.html`:
  - Formulario para subir **PDF** y (opcional) ingresar **URL** de red social.
  - Botón **Analizar datos** (estilo unificado).
- `result.html`:
  - Tarjeta con datos extraídos del PDF.
  - **Score** a la izquierda y **Riesgo** a la derecha (semáforo).
  - **Chart.js** con variables del modelo (si hubo clasificación).
  - Mensajes de **No PYME** / **No expediente** en formato destacado.
  - Botón **Nueva consulta**.

---

## Consideraciones

- El parseo del PDF asume el formato **oficial**; cambios futuros en el certificado podrían requerir ajustar las expresiones regulares pero siempre y cuando hayan cambios completos ala formato.
- `predict_proba` no está disponible (SVC sin `probability=True`); si se requiere probabilidad por clase, reentrenar el modelo con esa bandera.
- Para resultados coherentes, la columna **expediente** debe ser **int64** en los CSV y se debe garantizar la **misma normalización** que generó `df_score.csv`.

---

## Licencia
MIT (prototipo de research/POC).
