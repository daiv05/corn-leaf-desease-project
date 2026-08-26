# Interpretabilidad y Explicabilidad (XAI)

Una CNN es por defecto una caja negra, nos entrega una etiqueta pero no nos dice en qué se fijó para llegar a esa decisión.

Existen tres técnicas de explicabilidad más usadas en este campo: LIME, SHAP y Grad-CAM, primero veremos un poco de teoría para después justificar cuáles adoptamos en este proyecto.

¿Por qué es importante la explicabilidad en modelos de deep learning?

Una clasificación equivocada se puede traducir en una decisión de manejo equivocada, como aplicar el fertilizante que no era o fumigar sin necesidad, y por eso, tal como vimos con la brecha de dominio en los trabajos previos, no es suficiente con que el modelo acierte, necesitamos poder revisar que realmente está mirando dónde está el problema.

Estas herramientas en general se pueden diferenciar en su aplicación y en que tipo de explicación producen.

En su aplicación:
- LIME y SHAP son agnósticos al modelo, tratan a la red como una función de caja negra y solo necesitan poder consultarla.
- Grad-CAM en cambio es un método que aprovecha la estructura interna de la CNN, sus mapas de activación y sus gradientes, y por lo tanto solo sirven para esa familia de modelos.

En su tipo de explicación:
- Explicaciones locales, que justifican una predicción concreta, del tipo "¿por qué esta hoja fue clasificada como roya?".
- Y explicaciones globales, que describen el comportamiento del modelo en promedio sobre todo el conjunto. 

Las tres técnicas que se han decidido aplicar son fundamentalmente locales, aunque SHAP se puede agregar para aproximar una visión global, todas producen atribuciones, es decir un valor de importancia por cada componente de la entrada, ya sea un píxel, un superpíxel o una región espacial, que nos indica cuánto empujó ese componente hacia la clase predicha.

## LIME

LIME explica una predicción individual aproximando localmente el modelo complejo con un modelo lineal interpretable.

Para imágenes, LIME no trabaja con píxeles sueltos sino con superpíxeles, que son regiones contiguas de píxeles parecidos obtenidas por segmentación. El proceso segmenta la imagen en esos superpíxeles, genera muchas variantes perturbadas de la imagen original apagando algunos superpíxeles, le pregunta al modelo la probabilidad de la clase de interés para cada variante, y con esas respuestas ajusta un modelo lineal cuyos coeficientes son la importancia de cada superpíxel. En este proyecto se plantea usar mil perturbaciones por imagen y reportar los cinco superpíxeles más influyentes.

El resultado que se busca es un mapa donde los superpíxeles que sostienen la predicción se resaltan en un extremo del colormap divergente y los que la contradicen en el otro (aquí, rojo a favor y azul en contra: se evita el eje rojo-verde porque el verde ya significa tejido sano en la propia imagen y ese eje colapsa en daltonismo). Una explicación sana muestra la importancia concentrada sobre el tejido con síntomas, y una señal de alerta es que se disperse hacia el fondo o hacia el borde de la foto.

Entre las ventajas de usar LIME está que es agnóstico al modelo, muy intuitivo visualmente, y que produce importancias con signo, distinguiendo lo que va a favor de lo que va en contra. 

Entre sus desventajas podemos mencionar dos cosas: es el más costoso de los tres porque requiere alrededor de mil inferencias por imagen, y sobre todo es el más inestable, ya que al ser las perturbaciones aleatorias, dos ejecuciones con semillas distintas pueden dar explicaciones diferentes. Aunque para medir precisamente esto es que en el proyecto se incluirá una auditoría de estabilidad que correrá varias semillas y comparará los mapas resultantes.

## SHAP

SHAP le asigna a cada característica una importancia basada en los valores de Shapley, un concepto que viene de la teoría de juegos cooperativos. La idea es tratar cada superpíxel como si fuera un jugador y repartir de forma justa el crédito de la predicción entre todos ellos, promediando la contribución que aporta cada uno cuando se suma a las distintas combinaciones posibles de los demás.

Su gran atractivo teórico es que es el único método de atribución que cumple a la vez un conjunto de propiedades deseables, como la consistencia y que las características irrelevantes reciban importancia cero. En la práctica se usan aproximaciones, ya sea KernelSHAP, que es agnóstico al modelo y se puede ver como una versión con mejor fundamento teórico de LIME, o variantes como DeepSHAP y GradientSHAP, que son específicas para redes neuronales y mucho más rápidas porque aprovechan los gradientes. A diferencia de LIME, estas atribuciones son aditivas y consistentes, se pueden comparar entre imágenes distintas, y se pueden agregar sobre muchas imágenes para obtener una lectura global, por ejemplo qué regiones importan sistemáticamente para la clase de roya común. Su costo es alto en la variante KernelSHAP, aunque menor que el de LIME, y las variantes rápidas para redes profundas dependen de la arquitectura y de una elección cuidadosa de la línea base, que introduce sus propias suposiciones.

![SHAP vs LIME](/xai/shap_vs_lime.png)

## Grad-CAM

Grad-CAM es una técnica específica de CNN que produce un mapa de calor de baja resolución señalando qué regiones espaciales de la imagen activaron más la clase predicha. Toma la última capa convolucional, que es la de mayor carga semántica antes del pooling global y que todavía conserva estructura espacial, retropropaga el gradiente del logit de la clase objetivo hasta esos mapas de activación, promedia ese gradiente sobre el espacio para saber cuánto pesa cada mapa, y combina los mapas ponderados aplicando una ReLU para quedarse solo con la evidencia a favor, produciendo un mapa que después se sobremuestrea al tamaño de la imagen. 

En este proyecto se buscará realizar una implementación con hooks nativos de PyTorch y la capa objetivo de cada arquitectura estará mapeada explícitamente en el código para asegurar que se use la capa correcta.

La lectura será directa: se superpone el mapa de calor sobre la imagen y las zonas rojas son las de mayor influencia en la predicción mientras que las azules son las de menor, de modo que estamos viendo dónde mira el modelo. Es la técnica que mejor responde a la pregunta central de la explicabilidad en imágenes.

Sus ventajas son que es órdenes de magnitud más rápida, con una sola pasada hacia adelante y hacia atrás, que es determinista y por lo tanto no arrastra el problema de estabilidad de LIME, y que resulta muy legible para un experto de dominio. 

Como desventaja tiene que solo aplica a CNN, que su resolución está limitada por el mapa de activación de la última capa convolucional, con lo que localiza regiones amplias pero no bordes finos.

## Comparativa

Las diferencias principales entre las tres técnicas son las siguientes:

| Criterio | LIME | SHAP | Grad-CAM |
|---|---|---|---|
| Tipo | Agnóstico al modelo | Agnóstico (KernelSHAP) o específico (DeepSHAP) | Específico de CNN |
| Alcance | Local | Local, agregable a global | Local |
| Base | Modelo lineal sustituto sobre perturbaciones | Valores de Shapley | Gradientes por activaciones |
| Granularidad | Superpíxeles con signo | Superpíxel o píxel con signo, consistente | Región espacial en mapa de calor |
| Costo relativo por imagen | Alto | Medio (KernelSHAP) | Muy bajo |
| Estabilidad | Baja, por la perturbación aleatoria | Alta, determinista | Alta, determinista |
| Fundamento teórico | Heurístico | Fuerte, axiomas de Shapley | Basado en gradientes |

Bajo esta investigación decidimos adoptar la pareja LIME más Grad-CAM como base de explicabilidad, en las primeras etapas y experimentos con baselines, y dejar SHAP para el pipeline principal y los análisis globales una vez que tengamos un modelo final estable.

El detalle de los primeros análisis está en [Interpretabilidad del pipeline de baselines](../pipeline-baselines/interpretabilidad.md).

## Referencias

<a id="ref-9"></a>[9] C. Molnar, *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*, 2nd ed. 2022. [Online]. Available: https://christophm.github.io/interpretable-ml-book/

<a id="ref-10"></a>[10] M. T. Ribeiro, S. Singh, y C. Guestrin, "'Why Should I Trust You?': Explaining the Predictions of Any Classifier," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining (KDD)*, San Francisco, CA, USA, 2016, pp. 1135–1144.

<a id="ref-11"></a>[11] S. M. Lundberg y S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, Long Beach, CA, USA, 2017, pp. 4765–4774.

<a id="ref-12"></a>[12] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, y D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, Venice, Italy, 2017, pp. 618–626.

<a id="ref-13"></a>[13] R. Islam, M. Hasan, et al., "Advancing malware imagery classification with explainable deep learning: A state-of-the-art approach using SHAP, LIME and Grad-CAM," *PLOS One*, vol. 20, 2025. [Online]. Available: https://doi.org/10.1371/journal.pone.0318542

<a id="ref-14"></a>[14] Autores varios, "Robust multiclass classification of crop leaf diseases using hybrid deep learning and Grad-CAM interpretability," *Scientific Reports*, vol. 15, 2025. [Online]. Available: https://doi.org/10.1038/s41598-025-14847-7
