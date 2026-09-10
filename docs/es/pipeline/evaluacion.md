# Evaluación Rigurosa y Métricas Finales

La fase de evaluación del pipeline principal valida los modelos sobre el subconjunto de prueba independiente (**`test.csv` con 5,015 imágenes** retenidas que no participaron en el entrenamiento ni en la búsqueda de hiperparámetros).

El protocolo asegura la reproducibilidad científica y evalúa tanto la capacidad diagnóstica como la equidad de desempeño en condiciones reales de campo.

---

## 1. Métricas Primarias y Secundarias

Dado el desbalance natural de patologías vegetales en el dataset, la métrica primaria oficial es el **Macro $F_1$-Score**:
$$\text{Macro } F_1 = \frac{1}{C} \sum_{c=1}^C F_{1, c}$$

Esta métrica pondera por igual a todas las clases, impidiendo que el alto rendimiento en clases mayoritarias (*Healthy*, *Northern Corn Leaf Blight*) oculte deficiencias en clases minoritarias (*Potassium Deficiency*).

### Tabla de Resultados Comparativos en Test (5,015 Muestras):

| Modelo / Ensamble | Macro $F_1$ | Accuracy | Macro Precision | Macro Recall | Weighted $F_1$ |
|---|:---:|:---:|:---:|:---:|:---:|
| **[ShuffleNet-V2-x1.0](./entrenamiento)** | 0.9330 | 0.9731 | 0.9388 | 0.9298 | 0.9728 |
| **[EfficientNet-B0](./entrenamiento)** | 0.9483 | 0.9797 | 0.9589 | 0.9400 | 0.9793 |
| **[Soft Voting Ensemble](./ensamble)** 🏆 | **`0.9507`** | **`0.9799`** | **`0.9582`** | **`0.9445`** | **`0.9796`** |

---

## 2. Evaluación Desagregada por Entorno (*Cross-Environment*)

Para descartar que los modelos aprendan a clasificar por artefactos de fondo (como mesas de laboratorio o fondos monocromáticos) en lugar de la sintomatología foliar, el conjunto de prueba se evalúa de forma desagregada:

| Entorno Fotográfico | Muestras Test | Macro $F_1$ (EfficientNet-B0) | Disparidad ($\Delta F_1$) | Impacto Dispar ($DIR$) |
|---|:---:|:---:|:---:|:---:|
| **Laboratorio (`lab`)** | 3,118 | **0.9620** | — | — |
| **Campo Real (`real`)** | 1,897 | **0.9340** | **0.0280** | **`0.9709`** ($\ge 0.80$ ✅) |

El ratio de impacto dispar ($DIR = 0.9709$) supera con creces el umbral regulatorio del 80%, certificando que el modelo retiene su poder predictivo ante fondos naturales con suelo, malezas y sombras.

---

## 3. Detección Fuera de Distribución (OOD - Mahalanobis)

Para evitar que el sistema emita diagnósticos con alta confianza errónea sobre hojas de otros cultivos o imágenes no relacionadas, la evaluación incorpora el **Detector Fuera de Distribución (OOD)** basado en la distancia de Mahalanobis sobre el espacio latente de la penúltima capa:

$$D_M(x) = \min_{c} \sqrt{(\phi(x) - \mu_c)^T \Sigma^{-1} (\phi(x) - \mu_c)}$$

Las muestras con $D_M(x) > \tau_{95\%}$ son catalogadas automáticamente como **"Indeterminadas / Fuera de Distribución"**, activando una recomendación de inspección presencial para el agricultor.

---

## 4. Conexión con los Módulos del Pipeline

* **[Optimización de Hiperparámetros (Optuna)](./optimizacion):** Selección bayesiana de la tasa de aprendizaje, pesos de clase y tamaño de lote.
* **[Entrenamiento de Producción](./entrenamiento):** Curvas de convergencia de EfficientNet-B0 y ShuffleNet-V2.
* **[Ensamble Multimodelo](./ensamble):** Agregación por Soft Voting y matriz de confusión final.
* **[Interpretabilidad (XAI)](./interpretabilidad):** Mapas de atención Grad-CAM y verificación de fidelidad.
