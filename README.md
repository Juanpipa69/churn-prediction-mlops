# Churn Prediction MLOps

Proyecto final del curso de MLOps (Especialización en Ciencia de Datos e IA).
Pipeline end-to-end para predecir la cancelación (churn) de clientes de una
compañía de telecomunicaciones: tracking de experimentos, orquestación,
despliegue y monitoreo sobre un modelo de clasificación.

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

- **Métrica técnica principal:** ROC-AUC ≥ 0.85 en el set de validación.
- **Métrica de negocio:** dado que el costo de *no detectar* a un cliente que
  se va (falso negativo) es mayor que el de contactar a alguien que no iba a
  irse (falso positivo), priorizamos **recall de la clase Churn ≥ 0.65**
  manteniendo una precisión razonable (≥ 0.6), ajustando el umbral de decisión
  en vez de usar el 0.5 por defecto.
- **Baseline actual:** Logistic Regression — ROC-AUC 0.92, recall Churn 0.42,
  precisión Churn 0.84. El ROC-AUC ya cumple el objetivo técnico; el recall
  todavía no cumple el objetivo de negocio y es el primer punto a mejorar al
  comparar modelos en la Fase 2 (tracking con MLflow).

## Alcance (MVP de esta primera fase)

- [x] Selección de dataset y problema
- [x] EDA inicial y baseline con Logistic Regression
- [x] Experiment tracking con MLflow (Fase 2)
- [ ] Pipeline orquestado con Prefect (Fase 3)
- [ ] API de predicción + Docker (Fase 4)
- [ ] Propuesta de monitoreo (Fase 5)
- [ ] Tests, linter y documentación final (Fase 6)

## Estructura del repositorio

churn-prediction-mlops/
├── README.md
├── pyproject.toml / uv.lock
├── src/
│ └── data/
│ └── fetch_data.py # descarga el dataset (ucimlrepo)
├── data/
│ └── raw/churn.csv # generado localmente, no versionado
└── notebooks/
├── 01_eda.ipynb
└── 02_baseline.ipynb


## Cómo ejecutar el proyecto

```bash
# 1. Instalar dependencias
uv sync

# 2. Descargar el dataset
uv run python src/data/fetch_data.py

# 3. Abrir los notebooks en orden
uv run jupyter lab notebooks/01_eda.ipynb
```

## Resultados del baseline

| Modelo | ROC-AUC | Precision (Churn) | Recall (Churn) |
|---|---|---|---|
| Logistic Regression | 0.92 | 0.84 | 0.42 |

## Resultados de experimentos (MLflow)

| Modelo | ROC-AUC | Precision (Churn) | Recall (Churn) |
|---|---|---|---|
| Logistic Regression | 0.92 | 0.84 | 0.42 |
| Random Forest (balanced) | 0.98 | 0.73 | 0.97 |
| **XGBoost (scale_pos_weight) — campeón** | 0.99 | 0.85 | 0.94 |

Experimentos trackeados con MLflow (backend SQLite local). El modelo XGBoost quedó
registrado en el Model Registry como `churn-prediction-model`, alias `champion`.

