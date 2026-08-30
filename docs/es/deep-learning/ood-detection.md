# Detección de imágenes fuera de dominio (OOD)

El clasificador es **closed-set**: siempre elige una de las 9 clases de hoja de maíz, incluso si la foto no es una hoja. Frente a una mano, el cielo o una captura de pantalla, el modelo igual produce logits y softmax igual entrega una distribución de probabilidad — nada en esa salida distingue "estoy seguro de que es roya" de "no tengo ni idea, pero tengo que responder algo". Esta página describe cómo se detecta ese segundo caso: un módulo aparte, corrido sobre las features internas del modelo, que estima si una imagen se parece lo suficiente al dominio de entrenamiento antes de confiar en el diagnóstico.

## Por qué no basta con el umbral de confianza

Ya existe un primer filtro barato: rechazar una predicción si la confianza top-1 softmax no supera cierto umbral, o si el margen entre las dos clases más probables es demasiado chico (ver `MIN_CONFIDENCE`/`MIN_MARGIN` en `imageTensor.ts` de la app). Es útil pero insuficiente por diseño: softmax normaliza sobre las 9 clases que el modelo conoce, así que una imagen completamente ajena al dominio puede *igual* activar una clase con confianza alta si sus estadísticas de bajo nivel (color, textura) se parecen por casualidad a las de una clase de entrenamiento. La confianza softmax mide qué tan segura está la red **dado que tiene que elegir entre 9 opciones**, no qué tan parecida es la imagen a algo que la red haya visto.

Para eso hace falta mirar más adentro de la red: el vector de *features* antes de la capa de clasificación, y preguntar qué tan lejos cae ese vector de la región del espacio de features donde vive el dataset de entrenamiento.

## Mahalanobis distance (Lee et al. 2018)

El método base <sup>[[1]](#ref-1)</sup> asume que el vector de features pooled (penúltima capa, antes del head de clasificación) sigue, para cada clase, una distribución gaussiana multivariada con una **covarianza compartida entre clases** (más robusta de estimar que una covarianza por clase):

$$\hat\mu_c = \frac{1}{N_c}\sum_{i:\,y_i=c} f(x_i), \qquad \hat\Sigma = \frac{1}{N}\sum_{c}\sum_{i:\,y_i=c} (f(x_i) - \hat\mu_c)(f(x_i) - \hat\mu_c)^\top$$

donde $f(x_i)$ es el vector de features pooled de la muestra $i$. El score de una imagen nueva es la distancia de Mahalanobis a la clase más cercana:

$$M(x) = \min_c \; (f(x) - \hat\mu_c)^\top \hat\Sigma^{-1} (f(x) - \hat\mu_c)$$

Un $M(x)$ grande dice "este vector de features no se parece a ninguna clase conocida" — la señal que la confianza softmax no puede dar. `compute_ood_stats.py` calcula estas estadísticas sobre el split de train y `TFLiteInferenceEngine.ts`/`imageTensor.ts` las evalúan en el teléfono, con las features que produce la segunda salida del `.tflite` (ver [App React Native](../deployment/react-native.md)).

## Relative Mahalanobis Distance (RMD): el punto ciego de las dimensiones no discriminantes

$M(x)$ solo, sin más, falla en *near-OOD*: Ren et al. 2021 <sup>[[2]](#ref-2)</sup> muestran que la suma recorre **todas** las dimensiones del feature vector, y la mayoría codifican estadísticas genéricas de "imagen natural" (bordes, contraste, distribución de color) sin relación con si la imagen es una hoja de maíz. Como $\hat\Sigma^{-1}$ pondera cada dimensión por el inverso de su varianza, esas dimensiones no discriminantes —muchas más que las pocas que sí importan— **ahogan** la señal relevante.

El fix: ajustar una gaussiana adicional **sin condicionar por clase** — media $\mu_0$ y covarianza $\Sigma_0$ sobre todo el split de train, ignorando las etiquetas — y restar su distancia:

$$M_0(x) = (f(x)-\mu_0)^\top \Sigma_0^{-1} (f(x)-\mu_0), \qquad \mathrm{RMD}(x) = \min_c\, M_c(x) - M_0(x)$$

En una dimensión que no discrimina entre clases, la media y la varianza bajo el modelo condicionado a la clase son —por definición de "no discriminante"— prácticamente iguales a las del modelo de fondo, así que esa dimensión contribuye casi lo mismo a $M_c(x)$ y a $M_0(x)$ y se **cancela casi exactamente** al restar. Lo que sobrevive es la señal de las dimensiones donde el modelo de clase sí difiere del modelo de fondo. En el benchmark de Ren et al. (CIFAR-100 vs. CIFAR-10) esto sube el AUROC de 74.98 % a 81.08 %, sin introducir ningún hiperparámetro nuevo.

`tests/pipeline/test_compute_ood_stats.py::test_rmd_cancels_dimensions_shared_between_class_and_background_models` (y su espejo en TypeScript, `imageTensor.test.ts`) verifican esta cancelación con un ejemplo numérico: una dimensión "ruido" que domina $M_c(x)$ por sí sola desaparece por completo del score RMD.

## Por qué hace falta reducir dimensionalidad con PCA antes de ajustar las gaussianas

El feature vector pooled de `efficientnet_lite0` tiene 1280 dimensiones, pero su espectro de varianza real es muy sesgado: solo alrededor de un centenar concentran señal genuina, el resto es esencialmente ruido numérico de la red (activaciones casi constantes o altamente correlacionadas entre sí). Ajustar $\hat\Sigma$/$\Sigma_0$ directamente sobre las 1280 dimensiones crudas dejaba esas dimensiones de ruido con autovalores casi nulos; incluso regularizando antes de invertir, el "piso" de regularización en esas ~1100 dimensiones terminaba aportando la mayoría de la energía de la matriz inversa, con un peso miles de veces mayor que las dimensiones realmente informativas — la distancia de Mahalanobis quedaba dominada por ruido, no por señal.

La solución estándar (y la que usa este pipeline) es proyectar el feature vector L2-normalizado a las componentes principales que explican el 99 % de la varianza **antes** de calcular centroides, covarianza pooled y gaussiana de fondo:

$$\tilde f(x) = f(x) / \lVert f(x) \rVert_2 \qquad\text{(L2-normalización, Mahalanobis++, Ren et al. 2025: arXiv:2505.18032)}$$
$$z(x) = W\,(\tilde f(x) - \mu_{\text{pca}}) \qquad\text{(proyección a las top-}k\text{ componentes principales, }k\approx 186\text{ para efficientnet\_lite0)}$$

Todas las cantidades de la sección anterior ($\hat\mu_c$, $\hat\Sigma$, $\mu_0$, $\Sigma_0$) se calculan sobre $z(x)$, no sobre $f(x)$. `_fit_pca()`/`_apply_pca()` en `compute_ood_stats.py` implementan esto; `projectPCA()` en `imageTensor.ts` hace la misma proyección en el teléfono, con la matriz $W$ y la media $\mu_{\text{pca}}$ que vienen en `ood_stats.json`.

Ambas covarianzas, ya en el espacio reducido, se regularizan con un ridge proporcional a su traza antes de invertir (`_regularize_covariance()`):

$$\Sigma_{\text{reg}} = \Sigma + \lambda \cdot \frac{\mathrm{tr}(\Sigma)}{k}\, I, \qquad \lambda = 10^{-3}$$

## Calibración del umbral

El umbral de rechazo se calibra sobre el split de **val** (no visto durante el entrenamiento), como el percentil 95 del score RMD de cada muestra respecto al centroide de **su propia clase verdadera** — no el mínimo entre todas las clases que usa la inferencia. La razón de esta asimetría: en calibración conocemos la etiqueta real, así que medimos qué tan lejos cae genuinamente una muestra in-distribution de su propio centroide (el "radio" natural de esa clase); en inferencia no conocemos la etiqueta, así que usamos el mínimo sobre todas las clases como la mejor estimación disponible de esa misma cantidad. Para una muestra in-distribution bien clasificada, ambas coinciden.

Se usa el percentil 95 y no el 99 porque la cola de distancias de val puede tener outliers extremos (imágenes atípicas o mal etiquetadas) que inflan el percentil 99 muy por encima de donde vive la mayoría de los datos legítimos.

### Limitación conocida

Esta calibración usa **solo** datos in-distribution: no mide qué fracción de imágenes genuinamente fuera de dominio quedan por debajo del umbral (la métrica estándar en la literatura, FPR en el punto de operación TPR=95 %, requiere un conjunto de validación con ejemplos OOD reales <sup>[[1]](#ref-1)</sup><sup>[[2]](#ref-2)</sup>). El proyecto no cuenta hoy con un conjunto curado de imágenes OOD reales (fotos que no son hojas de maíz, con diversidad representativa de lo que un usuario podría fotografiar por error) — construirlo y recalibrar contra FPR95/AUROC real es el trabajo pendiente más directo para esta sección.

## Validación

`scripts/checks/validate_ood_detector.py` corre el detector calibrado contra tres grupos de imágenes: cinco sondas sintéticas claramente fuera de dominio (negro, blanco, gris sólido, ruido RGB aleatorio, azul sólido), una muestra de imágenes legítimas del split de test (para medir falsos positivos), y un grupo de imágenes reales sueltas (`--extra-images`) — una foto real de `data/clean/` por cada una de seis clases distintas, más una foto real que no es una hoja de maíz (un gráfico).

Resultado sobre `efficientnet_lite0` (`threshold=31.47`, percentil 95 del score RMD sobre val):

```
=== Imagenes sinteticas fuera de dominio ===
  black          RMD=  637.93  OOD (correcto)
  white          RMD=  370.94  OOD (correcto)
  gray_solid     RMD=  630.09  OOD (correcto)
  random_noise   RMD=  679.82  OOD (correcto)
  blue_solid     RMD=  620.43  OOD (correcto)
OOD sinteticas detectadas: 5/5

=== Muestra de imagenes legitimas de test ===
  Evaluadas: 200, falsos positivos: 7 (3.5%)

=== Imagenes reales sueltas ===
  healthy                    -> predicho=healthy                    RMD=  3.90  id
  common_rust                -> predicho=common_rust                RMD= -8.14  id
  gray_leaf_spot              -> predicho=gray_leaf_spot            RMD=  9.22  id
  northern_corn_leaf_blight   -> predicho=northern_corn_leaf_blight  RMD=  4.72  id
  fall_armyworm               -> predicho=fall_armyworm              RMD=  9.01  id
  nitrogen_deficiency         -> predicho=nitrogen_deficiency        RMD=-25.74  id
  foto real, no es hoja       -> predicho=fall_armyworm (sin sentido) RMD= 36.21  OOD (correcto)
```

Las cinco sondas sintéticas quedan marcadas como fuera de dominio, el falso-positivo sobre datos legítimos se mantiene cerca del 5 % nominal del percentil de calibración (3.5 %), las seis fotos reales de hoja (una por clase) se clasifican en su clase correcta y ninguna se marca como OOD, y la foto real que no es una hoja de maíz sí queda correctamente marcada como fuera de dominio.

## El contrato `ood_stats.json`

`compute_ood_stats.py` escribe `<run_dir>/export/ood_stats.json` (schema_version 4):

| Campo | Contenido |
|---|---|
| `feature_dim` | Dimensión del vector de features pooled crudo (1280 en `efficientnet_lite0`) |
| `pca_dim` | Dimensión tras la reducción PCA (`k`, ~186 para `efficientnet_lite0`) |
| `explained_variance` | Fracción de varianza realmente retenida con `pca_dim` componentes (≥ 0.99) |
| `l2_normalized` | `true` — recuerda al consumidor que debe normalizar antes de proyectar/comparar |
| `pca_mean_b64` | Media $\mu_{\text{pca}}$ usada al centrar antes de proyectar (feature_dim,) |
| `pca_components_b64` | Matriz de proyección $W$ (pca_dim × feature_dim), aplanada row-major |
| `mean_per_class_b64` | Centroides por clase $\hat\mu_c$ (en el espacio reducido), float32 binario en base64 |
| `inv_covariance_b64` | $\hat\Sigma^{-1}$ regularizada (pca_dim × pca_dim), aplanada row-major |
| `background_mean_b64` | $\mu_0$, la media de la gaussiana de fondo (en el espacio reducido) |
| `background_inv_covariance_b64` | $\Sigma_0^{-1}$ regularizada (pca_dim × pca_dim) |
| `threshold` | Percentil 95 del score RMD sobre val |

Los arrays van en binario float32 codificado en base64, no como JSON de texto, por tamaño. `sync_mobile_model.py` copia este archivo junto al `.tflite` y `labels.json` hacia `maize-doctor-app/assets/model/`, y avisa si falta.

El lado app (`TFLiteInferenceEngine.ts` / `imageTensor.ts`) decodifica estos campos y calcula `relativeMahalanobisDistance()` sobre la segunda salida del modelo (features pooled crudas, sin normalizar ni proyectar — `l2Normalize()` y `projectPCA()` se aplican dentro de la misma función, en ese orden, antes de comparar contra los centroides). Si el score RMD supera `threshold`, la predicción se marca `isUnrecognized`, igual que si fallara el chequeo de confianza/margen del softmax.

## Regenerar las estadísticas

```bash
# Requiere el mismo checkpoint que se exportó a .tflite
make compute-ood-stats MAIN_MODELS=efficientnet_lite0

# Sincronizar a la app (incluye ood_stats.json junto al .tflite y labels.json)
make sync-mobile-model RUN_DIR=outputs/main/efficientnet_lite0/<run_id> \
  DEST=../maize-doctor-app/assets/model

# Verificar con sondas sinteticas + imagenes reales antes de dar por buena la calibracion
python scripts/checks/validate_ood_detector.py --model efficientnet_lite0 \
  --checkpoint outputs/main/efficientnet_lite0/<run_id>/best.pth \
  --ood-stats outputs/main/efficientnet_lite0/<run_id>/export/ood_stats.json \
  --extra-images foto_hoja_real.jpg:legit foto_no_hoja.jpg:ood
```

El export a TFLite de dos salidas depende de `litert-torch`, que solo soporta Linux — este paso, igual que `export-main` con `EXPORT_FORMATS=tflite`, debe correr en WSL u otro entorno Linux (ver [App React Native](../deployment/react-native.md)).

## Referencias

<a id="ref-1"></a>[1] K. Lee, K. Lee, H. Lee, y J. Shin, "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 31, 2018.

<a id="ref-2"></a>[2] J. Ren, S. Fort, J. Liu, A. G. Roy, S. Padhy, y B. Lakshminarayanan, "A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection," *arXiv preprint arXiv:2106.09022*, 2021.

<a id="ref-3"></a>[3] M. Kirchhof, "Mahalanobis++: Improving OOD Detection via Feature Normalization," *arXiv preprint arXiv:2505.18032*, 2025.

<a id="ref-4"></a>[4] K. Kirchheim, M. Filax, y F. Ortmeier, "PyTorch-OOD: A Library for Out-of-Distribution Detection based on PyTorch," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2022. [Online]. Available: https://pytorch-ood.readthedocs.io/
