# Diseño de Monitoreo — Churn Prediction

Este documento propone la estrategia de monitoreo para el modelo `churn-prediction-model` una vez desplegado en producción (API FastAPI / Docker).

## 1. Métricas de negocio

Además de las métricas de modelo (ROC-AUC, recall, precision), en producción se debe vigilar:

- **Tasa de churn detectada vs. tasa de churn real**: cada mes, comparar cuántos clientes predichos como "churn" realmente cancelaron el servicio.
- **Costo de falsos negativos**: un cliente que iba a cancelar y no fue detectado (no recibe oferta de retención) representa pérdida de ingreso recurrente.
- **Costo de falsos positivos**: un cliente que no iba a cancelar pero fue marcado como riesgo (se gasta presupuesto de retención innecesariamente).
- **Volumen de predicciones diarias/semanales** vía el endpoint `/predict`, como indicador de uso real del servicio.

## 2. Data drift y model drift

Se propone usar **Evidently AI** para comparar la distribución de los datos que llegan en producción contra los datos de entrenamiento (`data/raw/churn.csv`).

**Features prioritarias a vigilar** (por ser las más predictivas según la importancia observada en el modelo XGBoost):

- `Customer Value`
- `Seconds of Use`
- `Complains`
- `Frequency of use`
- `Status`

**Frecuencia propuesta**: generar un reporte de drift semanal, comparando los últimos 7 días de predicciones (`current`) contra el dataset de entrenamiento (`reference`).

**Prueba de concepto**: en `scripts/generate_drift_report.py` se implementa un ejemplo funcional que compara el split de entrenamiento contra el split de prueba (simulando "producción"), y genera un reporte HTML con Evidently.

## 3. Criterio de reentrenamiento

Se dispara un reentrenamiento del modelo cuando ocurra **cualquiera** de las siguientes condiciones:

1. El **ROC-AUC** calculado sobre las predicciones de las últimas 2 semanas (con datos etiquetados reales) cae por debajo de **0.85** (el umbral objetivo definido en el README).
2. El **drift score global** reportado por Evidently supera el **50%** de features con drift detectado, sostenido durante **2 semanas consecutivas**.
3. Pasan **6 meses** desde el último entrenamiento, como límite de tiempo máximo aunque no se detecten señales de drift (mantenimiento preventivo).

## 4. Trabajo futuro (fuera de alcance de este proyecto)

- Automatizar la generación del reporte de drift como un flow de Prefect programado (ej. semanal).
- Alertas automáticas (Slack/email) cuando se cumpla el criterio de reentrenamiento.
- Dashboard de monitoreo en tiempo real con las métricas de negocio.