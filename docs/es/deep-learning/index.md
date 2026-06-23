# Deep Learning

Esta sección presenta los fundamentos teóricos que sustentan el pipeline de clasificación de enfermedades en hojas de maíz. Se parte de los conceptos básicos de redes neuronales convolucionales, se revisan las arquitecturas utilizadas en el proyecto y se discuten las estrategias para manejar el desbalance de clases propio del dataset.

---

## Redes Neuronales Convolucionales (CNN)

Una **red neuronal convolucional** (*Convolutional Neural Network*, CNN) es una clase de modelo de aprendizaje profundo diseñada para procesar datos con estructura de cuadrícula, como imágenes. A diferencia de las redes completamente conectadas, las CNNs explotan la localidad espacial y la invarianza a la traslación mediante tres operaciones clave <sup>[[15]](#ref-15)</sup>:

- **Capa convolucional:** aplica un conjunto de filtros aprendibles que convoluciona la entrada para producir mapas de activación. Cada filtro detecta un patrón local (bordes, texturas, formas) y comparte sus pesos en toda la imagen, reduciendo drásticamente el número de parámetros respecto a una capa densa.
- **Capa de *pooling*:** reduce la resolución espacial de los mapas de activación (por ejemplo, *max pooling* 2×2 con stride 2), introduciendo cierta invarianza a pequeñas traslaciones y reduciendo el costo computacional de las capas siguientes.
- **Capas completamente conectadas (*fully connected*):** al final de la red, transforman el vector de características extraídas por las capas convolucionales en una distribución de probabilidad sobre las clases mediante una función *softmax*.

### De LeNet a AlexNet

Las primeras CNNs modernas fueron propuestas por LeCun et al. en la década de 1990 para reconocimiento de dígitos escritos a mano. Sin embargo, fue la aparición de **AlexNet** en 2012 la que marcó el punto de inflexión del campo <sup>[[1]](#ref-1)</sup>: con 5 capas convolucionales y 3 capas FC entrenadas sobre ImageNet-1K con dos GPUs en paralelo, AlexNet redujo la tasa de error Top-5 de 26.2 % a 15.3 %, una brecha que convenció a la comunidad de que las CNNs profundas eran el camino.

AlexNet popularizó tres técnicas que se convirtieron en estándar:
- **ReLU** como función de activación, acelerando la convergencia frente a sigmoid/tanh.
- ***Dropout*** como regularización estocástica para reducir co-adaptación entre neuronas <sup>[[3]](#ref-3)</sup>.
- ***Data augmentation*** (recortes aleatorios, reflejos horizontales) para aumentar la variabilidad efectiva del conjunto de entrenamiento.

### Batch Normalization

Introducida por Ioffe y Szegedy en 2015, la **normalización por lotes** (*batch normalization*, BN) <sup>[[2]](#ref-2)</sup> normaliza las activaciones de cada capa a media cero y varianza unitaria sobre el mini-batch durante el entrenamiento, re-escalándolas luego con parámetros aprendibles $\gamma$ y $\beta$. Sus beneficios principales son:

- Permite usar tasas de aprendizaje más altas sin inestabilidad.
- Reduce la sensibilidad a la inicialización de pesos.
- Actúa como regularizador implícito, reduciendo la necesidad de dropout en algunas arquitecturas.

BN está presente en todas las arquitecturas modernas revisadas en este proyecto (EfficientNet, MobileNetV3, ResNet).

### Convoluciones Depthwise-Separable

Las **convoluciones depthwise-separable** <sup>[[5]](#ref-5)</sup> factorizan una convolución estándar $k \times k$ con $C_{in}$ canales de entrada y $C_{out}$ de salida en dos operaciones secuenciales:

1. **Depthwise convolution:** aplica un filtro $k \times k$ independiente *por canal*, produciendo $C_{in}$ mapas de activación.
2. **Pointwise convolution:** combina los $C_{in}$ mapas con una convolución $1 \times 1$ para producir los $C_{out}$ mapas finales.

Este factorización reduce el número de operaciones en un factor de aproximadamente $1/C_{out} + 1/k^2$, que para $k=3$ y $C_{out}$ grande equivale a una reducción de ~8–9× en FLOPs, sin degradación significativa de la capacidad representacional. Es la base de la familia MobileNet y heredada en los bloques MBConv de EfficientNet.

---

## Aprendizaje por Transferencia (*Transfer Learning*)

El **aprendizaje por transferencia** consiste en reutilizar un modelo pre-entrenado en una tarea fuente (aquí, clasificación de 1 000 categorías en ImageNet) y adaptarlo a una tarea objetivo diferente (clasificación de 9 clases de enfermedades en hojas de maíz) <sup>[[17]](#ref-17)</sup>.

### Por qué funciona con ImageNet

Los modelos pre-entrenados en ImageNet aprenden una jerarquía de características: las primeras capas detectan bordes y texturas de bajo nivel (independientes del dominio), las capas intermedias detectan partes y patrones visuales, y las capas más profundas codifican conceptos semánticos específicos de las categorías de ImageNet. Para tareas de visión en dominios visualmente relacionados (como la clasificación de hojas), los mapas de características de las capas bajas e intermedias son altamente reutilizables.

Esto produce beneficios concretos:
- **Convergencia más rápida:** los pesos ya son una buena inicialización, no un punto aleatorio.
- **Mejor generalización con pocos datos:** las características pre-aprendidas reducen el riesgo de sobreajuste cuando el dataset objetivo es pequeño.
- **Menor costo computacional:** es posible congelar las capas tempranas y entrenar solo las últimas, reduciendo el número de parámetros a optimizar.

### Fine-Tuning en este proyecto

La estrategia adoptada es **fine-tuning de la capa clasificadora**: se preservan todos los pesos del *backbone* pre-entrenado en ImageNet y se reemplaza únicamente la última capa lineal (el clasificador) por una `nn.Linear(in_features, 9)` para las 9 clases del proyecto. El backbone completo se deja entrenable (*unfrozen*) con una tasa de aprendizaje reducida para evitar destruir las representaciones pre-aprendidas.

Este enfoque es especialmente adecuado aquí porque el dataset de campo contiene texturas y patrones visuales (nervaduras de hoja, manchas foliares, coloraciones) que comparten estructura de bajo nivel con las categorías de ImageNet.

---

## Arquitecturas Ligeras y Escalables

### Redes Residuales (ResNet)

He et al. <sup>[[4]](#ref-4)</sup> demostraron que añadir más capas a una CNN no garantiza mejor rendimiento: más allá de cierta profundidad, el error de entrenamiento *aumenta* (fenómeno de degradación). Su solución fue introducir **conexiones residuales** (*skip connections*): la salida de un bloque de capas se suma directamente a su entrada, obligando a la red a aprender la función residual $\mathcal{F}(x) = H(x) - x$ en lugar de la función completa $H(x)$.

$$x_{l+1} = \mathcal{F}(x_l, \{W_l\}) + x_l$$

Esta reformulación facilita el flujo de gradientes hacia capas tempranas (mitigando el desvanecimiento del gradiente) y permite entrenar redes de 50, 101 o 152 capas con convergencia estable. ResNet-50 es uno de los modelos candidatos en la etapa de experimentación completa de este proyecto.

### Familia MobileNet

La familia MobileNet fue diseñada progresivamente para maximizar la precisión bajo restricciones de latencia en dispositivos móviles:

| Versión | Año | Innovación principal |
|---|---|---|
| MobileNet V1 <sup>[[5]](#ref-5)</sup> | 2017 | Convoluciones depthwise-separable |
| MobileNet V2 <sup>[[6]](#ref-6)</sup> | 2018 | Bloques MBConv con *inverted residuals* y *linear bottleneck* |
| MobileNet V3 <sup>[[7]](#ref-7)</sup> | 2019 | Búsqueda de arquitectura (NAS) + SE blocks + hard-swish |

Los **bloques MBConv** de V2 invierten la lógica del bottleneck clásico: en lugar de comprimir primero y expandir después, *expanden* el número de canales con una convolución $1 \times 1$, aplican la depthwise convolution en el espacio expandido, y luego *proyectan* de vuelta a un espacio de menor dimensión. La capa de proyección no tiene activación no lineal (de ahí *linear bottleneck*) para preservar la información en el espacio comprimido. V3 añade SE blocks y reemplaza ReLU6 con hard-swish en las capas más profundas.

### EfficientNet y *Compound Scaling*

Tan y Le <sup>[[8]](#ref-8)</sup> observaron que escalar profundidad, ancho y resolución de forma conjunta y equilibrada produce mejores resultados que hacerlo individualmente. Definieron un **coeficiente compuesto** $\phi$ tal que:

$$\text{depth} \propto \alpha^\phi, \quad \text{width} \propto \beta^\phi, \quad \text{resolution} \propto \gamma^\phi$$

con $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (restricción de recursos). Los valores $\alpha = 1.2$, $\beta = 1.1$, $\gamma = 1.15$ se determinaron por búsqueda en la red base B0 (encontrada con NAS). Aplicando $\phi = 1, 2, \ldots, 7$ se obtienen B1–B7.

EfficientNet-B0 alcanza ~77.1 % Top-1 en ImageNet con solo 5.3 M de parámetros, comparable a arquitecturas mucho más grandes de generaciones anteriores.

### ConvNeXt: CNNs Modernas Inspiradas en ViT

Liu et al. <sup>[[10]](#ref-10)</sup> partieron de ResNet-50 y aplicaron sistemáticamente las decisiones de diseño que hicieron exitosos a los Vision Transformers (tamaño de kernel más grande, capas normalizadas por capa en lugar de BN, activación GELU, menor frecuencia de downsampling). El resultado, **ConvNeXt**, es una CNN pura que supera a Swin Transformer en ImageNet con complejidad similar, sin mecanismos de atención.

Sus principales modificaciones respecto a ResNet son:
- Convoluciones $7 \times 7$ depthwise (mayor campo receptivo por bloque).
- *Patchify stem*: stem de $4 \times 4$ con stride 4, análogo al *patch embedding* de ViT.
- *Inverted bottleneck* y una sola capa de activación (GELU) por bloque.
- *Layer Normalization* (LN) en lugar de BN.

ConvNeXt-Tiny es uno de los modelos candidatos en la fase de experimentación completa del proyecto.

---

## Vision Transformers (ViT)

Dosovitskiy et al. <sup>[[11]](#ref-11)</sup> demostraron que un Transformer estándar aplicado directamente a parches de imagen puede igualar o superar a las CNNs en clasificación visual cuando se pre-entrena con suficientes datos. El **Vision Transformer (ViT)** divide una imagen de $H \times W$ píxeles en $N = \frac{H \cdot W}{P^2}$ parches no solapados de $P \times P$, proyecta cada parche a un vector de dimensión $D$ (*patch embedding*), añade codificación posicional aprendible y procesa la secuencia con un encoder Transformer estándar.

### Mecanismo de Auto-Atención Multi-Cabeza

El componente central del Transformer es la **atención de producto escalado**:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

donde $Q$, $K$, $V$ son proyecciones lineales de la secuencia de entrada. La **atención multi-cabeza** (*multi-head self-attention*, MHSA) ejecuta $h$ cabezas de atención en paralelo sobre subespacios distintos y concatena los resultados, permitiendo que el modelo atienda a múltiples relaciones posicionales y semánticas simultáneamente.

### Limitaciones con Datos Limitados

A diferencia de las CNNs, ViT carece de *inductive biases* de localidad y equivarianza a la traslación, por lo que requiere pre-entrenamiento en datasets de gran escala (JFT-300M, ImageNet-21K) para generalizar bien. Con datasets pequeños y transfer learning desde ImageNet-1K, las CNNs ligeras como EfficientNet-B0 o MobileNetV3 suelen superar a ViT-Base. Por ello, ViT se considera en este proyecto solo como arquitectura candidata en la fase de experimentación con el split completo (25 001 imágenes), no en la fase de baselines con el 20%.

---

## Manejo del Desbalance de Clases

El dataset de este proyecto presenta un desbalance severo: la clase más representada (*healthy*) supera en ~33× a la menos representada (*potassium_deficiency*). Sin intervención, un modelo puede alcanzar alta *accuracy* colapsando las predicciones hacia las clases mayoritarias.

### Función de Pérdida con Pesos de Clase

La **entropía cruzada ponderada** (*weighted cross-entropy*) asigna a cada clase $c$ un peso $w_c$ inversamente proporcional a su frecuencia <sup>[[12]](#ref-12)</sup>:

$$w_c = \frac{N}{C \cdot n_c}$$

donde $N$ es el total de muestras, $C$ el número de clases y $n_c$ la frecuencia de la clase $c$. La pérdida resultante es:

$$\mathcal{L} = -\sum_{c=1}^{C} w_c \cdot y_c \log \hat{p}_c$$

Esto hace que los errores en clases minoritarias contribuyan más al gradiente, empujando al modelo a aprender sus patrones.

### WeightedRandomSampler

El `WeightedRandomSampler` de PyTorch construye cada mini-batch muestreando ejemplos con probabilidad proporcional al inverso de la frecuencia de su clase. En combinación con la pérdida ponderada, iguala la frecuencia efectiva durante el entrenamiento sin duplicar muestras en memoria.

### Augmentation Extendido para Clases Minoritarias

Para las 5 clases con ratio de desbalance > 3.9× (`common_rust`, `gray_leaf_spot`, `nitrogen_deficiency`, `phosphorus_deficiency`, `potassium_deficiency`), el pipeline de `CornDataset` aplica un conjunto de augmentaciones más agresivo (rotaciones, color jitter, *random erasing*) que complementa el muestreo ponderado y reduce el riesgo de sobreajuste sobre las pocas muestras disponibles.

---

## Métricas de Evaluación

### Precisión, Recall y F1 por Clase

Para cada clase $c$ se calculan:

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}, \quad F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

### F1-Macro vs. F1-Weighted

El **F1-macro** promedia el F1 de cada clase con igual peso, sin importar cuántas muestras tenga:

$$F1_{\text{macro}} = \frac{1}{C} \sum_{c=1}^{C} F1_c$$

El **F1-weighted** pondera por la frecuencia de cada clase, sesgándose hacia las clases mayoritarias. En presencia de desbalance severo, F1-macro es la métrica más informativa porque penaliza igualmente el fallo en cualquier clase, incluidas las minoritarias de difícil diagnóstico agronómico.

**Umbral del proyecto:** F1-macro $\geq 0.85$ sobre el conjunto de prueba. Este umbral aplica tanto a los baselines (evaluados sobre el 20% del dataset) como a los modelos finales sobre el split completo.

---

## Deep Learning Aplicado a Enfermedades en Plantas

### Contexto del Problema

El uso de CNNs para el diagnóstico automático de enfermedades foliares fue popularizado por Mohanty et al. <sup>[[14]](#ref-14)</sup>, quienes entrenaron modelos sobre el dataset PlantVillage -imágenes de laboratorio con fondo uniforme- y reportaron precisiones superiores al 99 %. Sin embargo, estos resultados no se transfieren directamente a condiciones de campo reales: variaciones de iluminación, ángulo, fondo y resolución generan un *domain shift* que puede degradar drásticamente el rendimiento.

### Diferencia entre Imágenes de Laboratorio y Campo

El dataset de este proyecto incluye imágenes tanto de **laboratorio** (fondo controlado, iluminación uniforme) como de **campo real** (fondo natural, sombras, oclusiones parciales). Esta dualidad representa uno de los principales desafíos: el modelo debe aprender representaciones robustas al contexto de captura, no solo a los síntomas visuales de la enfermedad.

Las augmentaciones del pipeline de entrenamiento (`RandomResizedCrop`, `ColorJitter`, `RandomHorizontalFlip`, `GaussianBlur`) simulan parcialmente las variaciones de campo para reducir este *gap* de dominio.

### Trabajos Relevantes en Maíz

El diagnóstico de enfermedades en maíz con CNNs ha sido abordado en múltiples estudios recientes. Albahli y Masood (2022) reportaron 99.89 % de precisión en clasificación de enfermedades de maíz usando EfficientNetV2 sobre PlantVillage, lo que subraya el potencial de las arquitecturas de escala compuesta para este dominio <sup>[[8]](#ref-8)</sup>. Estudios con datos de campo real, no obstante, reportan resultados más conservadores (69–97 %) y destacan que la diversidad del entorno de captura es el principal factor de variabilidad.

El uso de modelos ligeros como MobileNetV3 y EfficientNet-B0 en lugar de arquitecturas más grandes (ResNet-101, EfficientNet-B7) responde a la restricción de despliegue del proyecto: un modelo TFLite de ≤ 20 MB con latencia ≤ 300 ms en dispositivos Android de gama media-baja.

---

## Referencias

<a id="ref-1"></a>[1] A. Krizhevsky, I. Sutskever, y G. E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 25, Lake Tahoe, NV, USA, 2012, pp. 1097–1105.

<a id="ref-2"></a>[2] S. Ioffe y C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift," in *Proc. 32nd Int. Conf. Mach. Learn. (ICML)*, Lille, France, 2015, pp. 448–456.

<a id="ref-3"></a>[3] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, y R. Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting," *J. Mach. Learn. Res.*, vol. 15, no. 1, pp. 1929–1958, 2014.

<a id="ref-4"></a>[4] K. He, X. Zhang, S. Ren, y J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Las Vegas, NV, USA, 2016, pp. 770–778.

<a id="ref-5"></a>[5] A. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, y H. Adam, "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications," *arXiv preprint arXiv:1704.04861*, Apr. 2017.

<a id="ref-6"></a>[6] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, y L.-C. Chen, "MobileNetV2: Inverted Residuals and Linear Bottlenecks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Salt Lake City, UT, USA, 2018, pp. 4510–4520.

<a id="ref-7"></a>[7] A. Howard, R. Pang, H. Adam, Q. V. Le, M. Sandler, B. Chen, W. Wang, L.-C. Chen, M. Tan, G. Chu, V. Vasudevan, y Y. Zhu, "Searching for MobileNetV3," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, Seoul, Korea, 2019, pp. 1314–1324.

<a id="ref-8"></a>[8] M. Tan y Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in *Proc. 36th Int. Conf. Mach. Learn. (ICML)*, Long Beach, CA, USA, 2019, pp. 6105–6114.

<a id="ref-9"></a>[9] J. Hu, L. Shen, y G. Sun, "Squeeze-and-Excitation Networks," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Salt Lake City, UT, USA, 2018, pp. 7132–7141.

<a id="ref-10"></a>[10] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, y S. Xie, "A ConvNet for the 2020s," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, New Orleans, LA, USA, 2022, pp. 11976–11986.

<a id="ref-11"></a>[11] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, y N. Houlsby, "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale," in *Proc. Int. Conf. Learn. Representations (ICLR)*, 2021.

<a id="ref-12"></a>[12] T.-Y. Lin, P. Goyal, R. Girshick, K. He, y P. Dollár, "Focal Loss for Dense Object Detection," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, Venice, Italy, 2017, pp. 2980–2988.

<a id="ref-13"></a>[13] Google Brain / TensorFlow Team, "Higher Accuracy on Vision Models with EfficientNet-Lite," *TensorFlow Blog*, Mar. 2020. [Online]. Available: https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html

<a id="ref-14"></a>[14] S. P. Mohanty, D. P. Hughes, y M. Salathé, "Using Deep Learning for Image-Based Plant Disease Detection," *Front. Plant Sci.*, vol. 7, p. 1419, Sep. 2016.

<a id="ref-15"></a>[15] R. Yamashita, M. Nishio, R. K. G. Do, y K. Togashi, "Convolutional Neural Networks: An Overview and Application in Radiology," *Insights Imaging*, vol. 9, no. 4, pp. 611–629, Aug. 2018.

<a id="ref-16"></a>[16] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, y L. Fei-Fei, "ImageNet: A Large-Scale Hierarchical Image Database," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Miami, FL, USA, 2009, pp. 248–255.

<a id="ref-17"></a>[17] S. J. Pan y Q. Yang, "A Survey on Transfer Learning," *IEEE Trans. Knowl. Data Eng.*, vol. 22, no. 10, pp. 1345–1359, Oct. 2010.
