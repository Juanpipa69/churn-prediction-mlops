# Churn Prediction MLOps

Proyecto final del curso de MLOps (Especialización en Ciencia de Datos e IA).
Pipeline end-to-end para predecir la cancelación (churn) de clientes de una
compañía de telecomunicaciones: tracking de experimentos, orquestación,
despliegue, monitoreo y buenas prácticas sobre un modelo de clasificación.

## Problema de negocio

Retener un cliente existente cuesta significativamente menos que adquirir uno
nuevo. Este proyecto busca anticipar qué clientes tienen mayor probabilidad de
cancelar el servicio, para que el equipo de retención pueda priorizar acciones
(ofertas, contacto proactivo) sobre ese segmento antes de perderlos.

## Dataset

- **Fuente:** [Iranian Churn Dataset](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset) (UCI Machine Learning Repository, id=563), obtenido vía la librería `ucimlrepo`.
- **Tamaño:** 3,150 clientes, 13 variables predictoras + variable objetivo `Churn` (binaria).
- **Variables:** indicadores de uso (llamadas fallidas, segundos de uso, frecuencia de uso y SMS, números distintos llamados), variables contractuales (duración de suscripción, plan tarifario, quejas, estado) y demográficas (edad, grupo etario), además de `Customer Value`.
- **Balance de clases:** ~84% no-churn / ~16% churn (desbalanceado).

## Objetivo y métrica de éxito

- **Métrica técnica principal:** ROC-AUC ≥ 0.85 en el set de validación. ✅ Cumplido (0.99 con el modelo campeón).
- **Métrica de negocio:** dado que el costo de *no detectar* a un cliente que
  se va (falso negativo) es mayor que el de contactar a alguien que no iba a
  irse (falso positivo), priorizamos **recall de la clase Churn ≥ 0.65**
  manteniendo una precisión razonable (≥ 0.6), ajustando el umbral de decisión
  en vez de usar el 0.5 por defecto. ✅ Cumplido (recall 0.94, precisión 0.85).
- **Modelo campeón:** XGBoost con `scale_pos_weight` para compensar el desbalance de clases (ver tabla de resultados abajo).

## Alcance del proyecto

- [x] Selección de dataset y problema
- [x] EDA inicial y baseline con Logistic Regression
- [x] Experiment tracking con MLflow (Fase 2)
- [x] Pipeline orquestado con Prefect (Fase 3)
- [x] API de predicción + Docker (Fase 4)
- [x] Propuesta de monitoreo (Fase 5)
- [x] Tests, linter y pre-commit (Fase 6)

## Estructura del repositorio

```
churn-prediction-mlops/
├── README.md
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .pre-commit-config.yaml
├── docs/
│   └── monitoring.md              # diseño de monitoreo (Fase 5)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_experiment_tracking.ipynb
├── scripts/
│   ├── export_model.py            # exporta el modelo campeón a models/champion/
│   └── generate_drift_report.py   # prueba de concepto de drift con Evidently
├── src/
│   ├── data/
│   │   ├── fetch_data.py          # descarga el dataset (ucimlrepo)
│   │   └── validate.py            # valida columnas, nulos y tamaño mínimo
│   ├── features/
│   │   └── preprocessing.py       # ColumnTransformer (imputación + escalado + one-hot)
│   ├── models/
│   │   └── train.py               # entrena, evalúa y loguea cada modelo en MLflow
│   ├── flows/
│   │   └── training_flow.py       # flow de Prefect: carga -> valida -> split -> entrena -> registra el campeón
│   └── api/
│       ├── main.py                # API FastAPI (/health, /predict)
│       └── schemas.py             # esquemas Pydantic de entrada/salida
├── tests/
│   ├── test_validate.py
│   ├── test_preprocessing.py
│   └── test_api.py
├── data/raw/churn.csv             # generado localmente, no versionado
├── models/champion/                # generado localmente, no versionado
└── reports/drift_report.html      # generado localmente, no versionado
```

## Cómo ejecutar el proyecto desde cero

```bash
# 1. Instalar dependencias
uv sync

# 2. Descargar el dataset
uv run python -m src.data.fetch_data

# 3. Entrenar, comparar modelos y registrar el campeón (orquestado con Prefect)
uv run python -m src.flows.training_flow

# 4. Exportar el modelo campeón a una carpeta autocontenida
uv run python scripts/export_model.py

# 5. Correr la API localmente
uv run uvicorn src.api.main:app --reload --port 8000
# -> documentación interactiva en http://127.0.0.1:8000/docs

# 6. O correr la API en Docker
docker build -t churn-prediction-api .
docker run -p 8000:8000 churn-prediction-api
```

## Calidad de código y tests (Fase 6)

- **Linter:** `ruff`, configurado en `pyproject.toml` (`[tool.ruff]`).
- **Tests unitarios:** `pytest`, en `tests/` — cubren validación de datos, preprocesamiento y la lógica de la API.
- **Pre-commit hooks:** `.pre-commit-config.yaml` — limpieza de espacios en blanco, fin de archivo, y `ruff` automático antes de cada commit.

```bash
uv run ruff check .                  # linter
uv run pytest -v                     # tests unitarios
uv run pre-commit run --all-files    # todos los hooks de pre-commit
```

Si clonas el repo por primera vez, instala los hooks de pre-commit una sola vez con:

```bash
uv run pre-commit install
```

## API de predicción

- `GET /health` — verifica que la API está viva y el modelo cargado.
- `POST /predict` — recibe los datos de un cliente y devuelve la predicción de churn y su probabilidad.

## Monitoreo (Fase 5)

El diseño completo de monitoreo está documentado en [`docs/monitoring.md`](docs/monitoring.md): métricas de negocio a vigilar, estrategia de detección de *data drift* con [Evidently AI](https://www.evidentlyai.com/) sobre las variables más predictivas, y el criterio de re-entrenamiento propuesto.

Como prueba de concepto, `scripts/generate_drift_report.py` genera un reporte real de drift comparando el split de entrenamiento contra el de prueba:

```bash
uv run python -m scripts.generate_drift_report
# -> reports/drift_report.html
```

## Resultados de experimentos (MLflow)

| Modelo | ROC-AUC | Precision (Churn) | Recall (Churn) |
|---|---|---|---|
| Logistic Regression | 0.92 | 0.84 | 0.42 |
| Random Forest (balanced) | 0.98 | 0.73 | 0.97 |
| **XGBoost (scale_pos_weight) — campeón** | 0.99 | 0.85 | 0.94 |

## Trabajo futuro

- Despliegue en la nube y pipeline de CI/CD (`.github/workflows/`).
- Automatizar la generación del reporte de drift como un flow de Prefect programado.
- Alertas automáticas cuando se cumpla el criterio de re-entrenamiento definido en `docs/monitoring.md`.

---

Este README se actualiza al cierre de cada fase del proyecto.
