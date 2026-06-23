# Baselines

Los baselines cumplen un doble propósito en este proyecto:

1. **Punto de comparación mínimo.** Cualquier arquitectura más compleja propuesta en etapas posteriores debe superar consistentemente estas cifras para justificar su mayor costo computacional o de mantenimiento.
2. **Demo de modelos candidatos.** Antes de comprometer el entrenamiento completo, los baselines permiten observar el comportamiento inicial de cada arquitectura candidata sobre una fracción representativa del dataset, detectar problemas (colapso de clases, overfitting temprano, incompatibilidad con el pipeline) a bajo costo.

Los tres modelos elegidos son redes convolucionales ligeras pre-entrenadas en ImageNet. Se priorizó este perfil porque el objetivo final es un sistema desplegable en entornos con recursos limitados (dispositivos móviles o cómputo en campo), y porque al tener parámetros comparables entre sí hacen que las diferencias de rendimiento sean atribuibles a la arquitectura, no al tamaño.

---

## Dataset utilizado

Los baselines se entrenan sobre un **20 % del dataset limpio**, muestreado de forma estratificada en `splits/seed_42_sample20/`:

| Split | Imágenes |
|---|---:|
| Entrenamiento (`train.csv`) | 4 426 |
| Validación (`val.csv`) | 949 |
| Prueba (`test.csv`) | 949 |
| **Total** | **6 324** |

El split completo (100 %, `splits/seed_42/`) contiene 25 001 imágenes en proporción 70 / 15 / 15. El 20 % mantiene esa misma proporción y la estratificación por `label + environment`, por lo que la distribución de clases y entornos es representativa del corpus completo.

**Por qué el 20 % y no el 100 %:** los baselines son una exploración inicial, no el entrenamiento definitivo. Con una fracción del corpus es posible comparar el comportamiento de las tres arquitecturas -detectar colapso de clases, gradientes muertos, diferencias de convergencia- en un tiempo de cómputo mucho menor, sin comprometer el ciclo completo de experimentación. Los modelos finalistas identificados aquí se re-entrenarán sobre el split completo en la fase de producción.

---

## Modelos seleccionados

### MobileNetV3-Large

**MobileNetV3-Large** es una red neuronal convolucional diseñada por Google para ejecutarse eficientemente en dispositivos móviles y de borde (*edge*). Publicada en 2019, combina tres innovaciones <sup>[[7]](#ref-7)</sup>:

- **Depthwise separable convolutions:** factoriza una convolución estándar en dos pasos (por canal, luego entre canales), reduciendo el número de operaciones en un factor de ~8–9× respecto a una convolución densa equivalente <sup>[[5]](#ref-5)</sup>.
- **Squeeze-and-Excitation (SE) blocks:** un mecanismo de atención de canal que aprende a recalibrar la importancia relativa de cada mapa de activación, mejorando la capacidad representacional sin aumentar los FLOPs en forma significativa <sup>[[9]](#ref-9)</sup>.
- **Hard-swish activation:** una aproximación lineal a la función Swish que es 15 % más rápida en hardware sin unidades de punto flotante sofisticadas.

La variante **Large** (vs. Small) maximiza la precisión dentro del espacio de modelos móviles. En ImageNet-1K alcanza ~75 % de Top-1 con solo 5.4 M de parámetros y ~219 M de MACs, haciéndola viable en CPU o GPU de gama baja.

**Trade-offs relevantes para este proyecto:**

| Aspecto | Detalle |
|---|---|
| Precisión | Inferior a EfficientNet-B0 en ~2–3 pp en ImageNet, pero suficiente para tareas de clasificación de hojas con dominio controlado |
| Velocidad de inferencia | Muy alta: diseñada explícitamente para latencia en móviles (Pixel 1: ~51 ms) |
| Tamaño del modelo | ~21 MB serializado; desplegable sin cuantización |
| Transfer learning | Pre-entrenada en ImageNet con `MobileNet_V3_Large_Weights.DEFAULT` (IMAGENET1K_V2); solo se reemplaza la última capa lineal del clasificador |
| Riesgo de underfitting | Puede ser insuficiente para distinguir clases con síntomas visuales similares (`gray_leaf_spot` vs. `northern_corn_leaf_blight`) |

En el código, se construye reemplazando `model.classifier[-1]` por una `nn.Linear(in_features, 9)` para las 9 clases del proyecto.

---

### EfficientNet-B0

**EfficientNet-B0** es la red base de la familia EfficientNet, propuesta por Google Brain en 2019. Su contribución central es el *compound scaling*: en lugar de escalar solo la profundidad, el ancho o la resolución de entrada de forma independiente (como hacía la práctica anterior), EfficientNet escala los tres simultáneamente con un coeficiente compuesto $\phi$ determinado por búsqueda de arquitectura (NAS) <sup>[[8]](#ref-8)</sup>.

La versión B0 es el punto de partida de la familia: la arquitectura base encontrada por NAS antes de aplicar cualquier escala adicional. Usa bloques **MBConv** (Mobile Inverted Bottleneck) <sup>[[6]](#ref-6)</sup> con:
- Conexiones residuales
- Expansión de canales seguida de proyección
- Squeeze-and-Excitation integrado en cada bloque <sup>[[9]](#ref-9)</sup>

En ImageNet-1K alcanza ~77.1 % de Top-1 con 5.3 M de parámetros -similar a MobileNetV3-Large en tamaño, pero con mayor precisión.

**Trade-offs relevantes para este proyecto:**

| Aspecto | Detalle |
|---|---|
| Precisión | ~2–3 pp superior a MobileNetV3-Large en ImageNet; diferencia esperada también en fine-tuning |
| Velocidad de inferencia | Más lento que MobileNetV3 en hardware móvil (~1.3–1.5× más latencia) por las operaciones SE en cada bloque |
| Tamaño del modelo | ~20 MB serializado; comparable a MobileNetV3 |
| Transfer learning | Pre-entrenado en ImageNet con `EfficientNet_B0_Weights.DEFAULT` (IMAGENET1K_V1); se reemplaza `model.classifier[1]` |
| Regularización implícita | Los bloques MBConv con dropout estructural hacen a EfficientNet-B0 más robusto al overfitting con datasets pequeños |

---

### EfficientNet-Lite0

**EfficientNet-Lite0** es una variante de EfficientNet-B0 optimizada específicamente para dispositivos de borde con aceleradores de inferencia (Coral Edge TPU, microcontroladores ARM con CMSIS-NN) <sup>[[13]](#ref-13)</sup>. Las diferencias con B0 son:

- **Sin Squeeze-and-Excitation:** los bloques SE se eliminan porque su operación de reducción global no se mapea eficientemente en aceleradores de inferencia cuantizados.
- **ReLU6 en lugar de Swish:** más compatible con cuantización INT8, donde Swish introduce errores de representación no triviales que hacen caer la precisión de ~75 % a ~46 % si no se sustituye <sup>[[13]](#ref-13)</sup>.
- **Sin stem strided convolution:** la red evita algunas operaciones que rompen la compatibilidad con ciertos compiladores de modelos (TFLite, ONNX para Edge TPU).

Se construye con `timm` (`timm.create_model("efficientnet_lite0", pretrained=True, num_classes=9)`) porque `torchvision` no incluye esta variante.

**Trade-offs relevantes para este proyecto:**

| Aspecto | Detalle |
|---|---|
| Precisión | ~1–2 pp inferior a EfficientNet-B0 en ImageNet (~74–75 % Top-1); compensado por facilidad de despliegue |
| Velocidad de inferencia | Similar o superior a MobileNetV3-Large en aceleradores compatibles; sin ventaja clara en GPU estándar |
| Cuantización | Diseñada para cuantizarse a INT8 sin degradación significativa; punto fuerte para despliegue en campo |
| Dependencia adicional | Requiere `timm` (no incluida en `torchvision`); añade una dependencia al entorno |
| Uso en proyecto | Representa el extremo del trade-off "máxima eficiencia en edge" para comparar contra el extremo "máxima precisión" de EfficientNet-B0 |

---

## Comparación de los tres modelos

| Modelo | Parámetros | Top-1 ImageNet | Latencia relativa | Apto para edge cuantizado |
|---|---:|---:|---|---|
| `mobilenet_v3_large` | 5.4 M | ~75.2 % | ★★★ (más rápido) | Parcialmente |
| `efficientnet_b0` | 5.3 M | ~77.1 % | ★★ | No (por SE + Swish) |
| `efficientnet_lite0` | 4.7 M | ~74.9 % | ★★★ | Sí (diseñado para eso) |

Los tres parten de pesos pre-entrenados en ImageNet <sup>[[16]](#ref-16)</sup> y se ajustan sobre el dataset de maíz con:
- `CrossEntropyLoss` con pesos de clase inversamente proporcionales a la frecuencia <sup>[[12]](#ref-12)</sup>
- `WeightedRandomSampler` para igualar la frecuencia efectiva durante entrenamiento
- Pipeline de augmentation extendido para las 5 clases minoritarias

El modelo con mejor F1-macro en el conjunto de prueba establecerá el **umbral de referencia** que las arquitecturas posteriores (ResNet-50, ConvNeXt, ViT) deberán superar de forma consistente.

---

## Referencias

<a id="ref-5"></a>[5] A. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, y H. Adam, "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," *arXiv preprint arXiv:1704.04861*, Apr. 2017.

<a id="ref-6"></a>[6] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, y L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Salt Lake City, UT, USA, 2018, pp. 4510–4520.

<a id="ref-7"></a>[7] A. Howard, R. Pang, H. Adam, Q. V. Le, M. Sandler, B. Chen, W. Wang, L.-C. Chen, M. Tan, G. Chu, V. Vasudevan, y Y. Zhu, "Searching for MobileNetV3," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, Seoul, Korea, 2019, pp. 1314–1324.

<a id="ref-8"></a>[8] M. Tan y Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in *Proc. 36th Int. Conf. Mach. Learn. (ICML)*, Long Beach, CA, USA, 2019, pp. 6105–6114.

<a id="ref-9"></a>[9] J. Hu, L. Shen, y G. Sun, "Squeeze-and-Excitation Networks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Salt Lake City, UT, USA, 2018, pp. 7132–7141.

<a id="ref-12"></a>[12] T.-Y. Lin, P. Goyal, R. Girshick, K. He, y P. Dollár, "Focal Loss for Dense Object Detection," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, Venice, Italy, 2017, pp. 2980–2988.

<a id="ref-13"></a>[13] Google Brain / TensorFlow Team, "Higher Accuracy on Vision Models with EfficientNet-Lite," *TensorFlow Blog*, Mar. 2020. [Online]. Available: https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html

<a id="ref-16"></a>[16] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, y L. Fei-Fei, "ImageNet: A Large-Scale Hierarchical Image Database," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Miami, FL, USA, 2009, pp. 248–255.
