# Propuesta de Perfil de Proyecto

**Deteccion de Enfermedades Foliares, Plagas y Deficiencias Nutricionales en Cultivos de Maiz mediante Deep Learning en Dispositivos Moviles**

- Universidad de El Salvador
- Facultad de Ingenieria y Arquitectura
- Escuela de Ingenieria de Sistemas Informaticos
- Curso de Especializacion en Machine Learning
- Ciclo I 2026
- Grupo de trabajo: 02
- Docente: Ing. Bladimir Diaz Campos

**Presentado por**

| Apellido       | Nombre         | Carnet |
| ---            | ---            | ---    |
| Rivas Fuentes  | Josias Abner    | RF20010 |
| Deras Cerros   | David Alejandro | DC19019 |
| Rosales Molina | Elmer Edenilson | RM20001 |

Ciudad Universitaria, 18 de abril de 2026

## Contenido

- Planteamiento del problema
- Relevancia social
- Objetivos
- Justificacion e impacto esperado
- Marco teorico
- Metodologia
- Cronograma
- Recursos necesarios
- Riesgos
- Bibliografia
- Anexos

## Planteamiento del problema

La agricultura en El Salvador es un pilar socioeconomico fundamental para la seguridad alimentaria y el desarrollo rural, representado aproximadamente el 5.6% del Producto Interno Bruto (PIB). El ecosistema agricola salvadoreno se distingue por una marcada fragmentacion de la tenencia de la tierra y una alta concentracion de pequenos productores. Los registros censales agropecuarios recientes establecen la existencia de aproximadamente 395,588 productores a nivel nacional. De este total, 325,044 individuos (representando el 82.1% de la demografia agricola) se clasifican como pequenos productores, muchos de los cuales operan a nivel de agricultura familiar o de estricta subsistencia.

Dentro de este sector, el cultivo del maiz forma parte de la economia agropecuaria y constituye uno de los principales alimentos de la dieta salvadorena y una fuente clave de ingresos para pequenos y medianos productores. Sin embargo, la produccion de maiz se ve afectada por diversos factores, entre los cuales destacan las enfermedades y plagas que reducen negativamente el rendimiento del cultivo. De acuerdo con la Organizacion de las Naciones Unidas para la Alimentacion y la Agricultura (FAO), hasta un 40% de los cultivos a nivel mundial se pierden debido a plagas y enfermedades, lo cual refleja la magnitud del problema tambien en el contexto salvadoreno.

A nivel nacional, los agricultores enfrentan dificultades para identificar de manera temprana las enfermedades que afectan sus cultivos de maiz, ya que el diagnostico suele basarse en la observacion visual y en conocimiento empirico. Esta situacion se agrava en zonas rurales donde el acceso a asistencia tecnica especializada es limitado, lo que provoca que las enfermedades sean detectadas en etapas avanzadas, reduciendo la productividad y generando perdidas economicas significativas.

En los ultimos anos, el acceso a las tecnologias de la informacion y comunicacion (TIC) se ha vuelto atractivo para el sector agropecuario para mejorar la eficiencia y productividad. Aunque la adopcion de estas tecnologias es limitada, especialmente para pequenos productores se considera el creciente acceso a telefonos inteligentes a nivel nacional y el avance de la inteligencia artificial, particularmente en el campo de la vision por computadora que ha permitido el desarrollo de modelos capaces de identificar enfermedades en plantas a partir de imagenes. A partir de estas novedades, surge la oportunidad de desarrollar soluciones accesibles que permitan a los agricultores identificar enfermedades en sus cultivos de maiz mediante el uso de imagenes.

Las enfermedades foliares del maiz, como los tizones y las royas, presentan sintomas visuales que pueden confundirse con otros factores como estres hidrico, dano por plagas o deficiencias nutricionales, dificultando la toma de decisiones oportunas. En este contexto, surge la necesidad de desarrollar un sistema basado en Machine Learning que permita clasificar enfermedades en cultivos de maiz a partir de imagenes, incorporando ademas recomendaciones generales que apoyen la toma de decisiones del agricultor.

Adicionalmente, se propone una vision de sistema evolutivo, implementado mediante herramientas portatiles, accesibles y funcionales incluso en entornos con conectividad limitada. Este sistema permitira la recoleccion de nuevas imagenes en condiciones reales de uso, con el objetivo de facilitar futuras mejoras del modelo mediante procesos de reentrenamiento con datos locales. Este enfoque busca contribuir a mitigar la limitada disponibilidad de datasets abiertos en el contexto salvadoreno, cuya escasez o fragmentacion puede afectar la robustez de modelos entrenados exclusivamente con imagenes obtenidas en condiciones controladas.

## Relevancia social

El desarrollo de este proyecto adquiere importancia debido a su relevancia en el sector agricola salvadoreno, especialmente en los pequenos productores que dependen del cultivo de maiz como principal fuente de sustento.

Contar con herramientas accesibles que faciliten el reconocimiento de enfermedades contribuye a mejorar las condiciones de produccion, favoreciendo una gestion mas eficiente de los cultivos. Asimismo, este tipo de soluciones puede apoyar la reduccion de perdidas, el aprovechamiento adecuado de recursos y la mejora en la calidad de las cosechas.

En conjunto, el proyecto se orienta a fortalecer las capacidades de los productores, contribuyendo al desarrollo del sector agropecuario y al bienestar de las comunidades vinculadas a esta actividad.

## Objetivos

### Objetivo general

Desarrollar e implementar un sistema movil basado en aprendizaje profundo capaz de clasificar 9 clases en hojas de maiz (5 enfermedades foliares y plagas, y 3 deficiencias nutricionales, mas hoja sana) a partir de imagenes capturadas con telefonos inteligentes, optimizado para ejecutarse localmente en dispositivos Android de gama media/baja, con inferencia completamente offline y funcionalidades opcionales de sincronizacion en linea.

### Objetivos especificos

1. Identificar, evaluar y consolidar datasets publicos de imagenes de hojas de maiz en condiciones controladas y de campo real, construyendo un repositorio curado de al menos 18,000 imagenes distribuidas en 9 clases, con un minimo de 2,000 imagenes de campo real por clase, documentando licencias y restricciones de uso, procedencia y dominio de captura, criterios de limpieza y estrategias para mitigar desbalance.
2. Entrenar un modelo liviano de clasificacion multiclase basado en una arquitectura liviana, utilizando transferencia de aprendizaje estableciendo como meta un Macro F1-score >= 0.85 en un conjunto de prueba completamente independiente y compuesto prioritariamente por imagenes reales de campo. La evaluacion incluira matriz de confusion y curvas Precision-Recall por clase.
3. Convertir el modelo entrenado a TensorFlow Lite aplicando cuantizacion post-entrenamiento (Int8), logrando un tamano del modelo de <= 20 MB y una latencia objetivo de <= 300 ms por inferencia en un dispositivo Android de gama media/baja (>=4GB RAM, CPU Snapdragon serie 6xx o equivalente).
4. Implementar una PWA que permita capturar o seleccionar imagenes, ejecutar inferencia local sin conexion a internet y mostrar la clase predicha junto con su nivel de confianza.
5. Incorporar un modulo opcional que permita almacenar y sincronizar imagenes junto con metadatos minimos (clase, fecha y ubicacion aproximada bajo consentimiento), documentando un protocolo de gobernanza y uso futuro de los datos, sin incluir reentrenamiento dentro del alcance del presente proyecto.

## Justificacion e impacto esperado

El maiz es el pilar de la seguridad alimentaria salvadorena, sosteniendo a mas de 2 millones de personas rurales con una produccion de 17.1 millones de quintales en 376,733 manzanas. Sin embargo, el rendimiento es muy bajo (45.5 quintales / manzana) y enfermedades foliares agresivas pueden destruir hasta el 70% de la cosecha sin una deteccion en las primeras dos semanas. El sector enfrenta su peor crisis en una decada: en 2023 la cosecha cayo un tercio en comparacion con 2021 debido a factores climaticos y altos costos de insumos. Los pequenos productores operan con un margen de error economico casi nulo, donde comprar el agroquimico equivocado o actuar tarde resulta devastador. Ante la falta de asistencia tecnica inmediata, es imperativo implementar una solucion tecnologica descentralizada. El despliegue de una herramienta basada en Edge AI no es solo un avance tecnologico, sino una medida urgente y de bajo costo para proteger la economia de los agricultores de subsistencia mediante diagnosticos precisos y oportunos.

## Marco teorico

### Redes Neuronales Convolucionales (CNN)

Las Redes Neuronales Convolucionales (CNN) son una arquitectura de Deep Learning disenada especificamente para el procesamiento de datos con estructura espacial, como las imagenes. Su principal fortaleza radica en la capacidad de aprender automaticamente representativas jerarquicas a partir de los datos sin necesidad de ingenieria manual de caracteristicas.

Las imagenes digitales pueden representarse como matrices tridimensionales:

$$
I(x, y) \in R^{h \times w \times c}
$$

donde $h$ corresponde a la altura, $w$ al ancho y $c$ al numero de canales (por ejemplo, RGB).

Las CNN aplican filtros (kernels) que recorren la imagen para detectar patrones relevantes mediante la operacion de convolucion:

$$
y(i, j) = \sum_m \sum_n x(i + m, j + n) \cdot w(m, n)
$$

- $x$: matriz de entrada (por ejemplo, los pixeles de una imagen).
- $w$: filtro o kernel (los pesos que la red aprende).
- $y(i, j)$: valor resultante en la posicion $(i, j)$ de la capa de salida.
- $m, n$: indices que recorren las dimensiones del filtro.

A traves de multiples capas, la red aprende diferentes niveles de abstraccion: capas iniciales para deteccion de bordes y texturas, capas intermedias para formas y estructuras y capas profundas para patrones complejos del dominio.

En el contexto agricola, estas caracteristicas incluyen los bordes y contornos de hojas, texturas superficiales, patrones necroticos y las distribuciones de manchas foliares.

### Transfer Learning

El Transfer Learning es una tecnica que permite aprovechar modelos previamente entrenados en grandes conjuntos de datos, como ImageNet, para resolver problemas especificos.

En este enfoque, se reutilizan los pesos de un modelo pre-entrenado y se ajustan sus ultimas capas al nuevo dominio:

$$
\theta = \theta_{pretrained} + \Delta \theta
$$

Esto permite que el modelo conserve conocimiento general sobre patrones visuales, como bordes y formas, y lo adapte a caracteristicas especificas como enfermedades en hojas.

### Funcion de perdida

La funcion de perdida mide el error entre las predicciones del modelo y los valores reales. Una de ellas es Categorical Cross-Entropy, adecuada para problemas de clasificacion multiclase:

$$
L = - \sum_{i=1}^n y_i \log(\hat{y}_i)
$$

Esta funcion penaliza fuertemente las predicciones incorrectas con alta confianza, favoreciendo una mejor calibracion del modelo.

Adicionalmente, se implementa una ponderacion de clases para abordar el desbalance del dataset, asignando mayor importancia a clases minoritarias, lo cual mejora la deteccion de enfermedades menos frecuentes.

### Optimizacion del modelo

El entrenamiento del modelo se realiza mediante algoritmos de optimizacion que ajustan los parametros para minimizar la funcion de perdida. Uno de los metodos mas utilizados es el gradiente descendente:

$$
\theta = \theta - \alpha \nabla J(\theta)
$$

Adam es otra tecnica y esta construido sobre el gradiente descendiente, combina tecnicas de momento y adaptacion de la tasa de aprendizaje, permitiendo una convergencia mas rapida y estable, permitiendo un entrenamiento mas rapido, fluido y fiable, especialmente para el aprendizaje profundo.

### Metricas de Evaluacion

Para evaluar el desempeno del modelo se utilizan las siguientes metricas:

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
Precision = \frac{TP}{TP + FP}
$$

$$
Recall = \frac{TP}{TP + FN}
$$

$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$

En este proyecto se prioriza el Recall, debido a la importancia de minimizar falsos negativos en la deteccion de enfermedades. La no deteccion de una enfermedad puede tener consecuencias criticas en la produccion agricola.

El uso del F1-Score permite equilibrar precision y sensibilidad, especialmente en datasets desbalanceados.

### Arquitectura del modelo

La arquitectura del modelo constituye la estructura interna que define la organizacion y conexion de las capas que conforman una red neuronal.

Arquitecturas eficientes como MobileNetV3, disenadas para operar en dispositivos con recursos computacionales limitados, permiten utilizar tecnicas como Transfer Learning. La estructura del modelo comprende una base convolucional preentrenada para la extraccion de caracteristicas, seguida de capas densas personalizadas para la adaptacion al problema especifico, y una capa de salida con funcion de activacion Softmax para la clasificacion multiclase. Este enfoque permite mantener un equilibrio entre precision predictiva y eficiencia computacional.

### Despliegue del modelo

El despliegue de modelos de aprendizaje profundo requiere una arquitectura que garantice eficiencia y tiempos de respuesta adecuados. En este proyecto se adopta un enfoque hibrido, utilizando FastAPI como backend para la gestion de solicitudes e inferencia, y un modelo optimizado con TensorFlow Lite para su ejecucion eficiente. TensorFlow Lite permite reducir el tamano del modelo y mejorar la latencia, facilitando su implementacion en dispositivos moviles.

Esta arquitectura permite flujos como:

1. Captura de imagen / input
2. Preprocesamiento
3. Inferencia mediante CNN
4. Generacion de prediccion
5. Respuesta al usuario

## Metodologia

La gestion y ejecucion del proyecto se desarrollara bajo la metodologia CRISP-DM (Cross-Industry Standard Process for Data Mining). Gracias al enfoque iterativo la metodologia permitira conforme se identifiquen mejoras, alinear las decisiones tecnicas del modelo con las necesidades reales del entorno agricola y las restricciones de hardware movil.

### Comprension del Negocio

El maiz es uno de los cultivos mas importantes en El Salvador, tanto para consumo interno como para la economia de pequenos y medianos productores.

Las enfermedades foliares como la roya comun, mancha gris o el tizon foliar pueden reducir la produccion si no se detectan de forma temprana, y es que en la practica el diagnostico suele depender de la propia experiencia empirica del agricultor, recomendaciones informales y tal vez visitas tecnicas del MAG u organizaciones locales, esto genera diagnosticos tardios o incorrectos, lo que impacta en perdidas economicas y uso ineficiente de insumos agricolas.

El desarrollo de una herramienta basada en vision por computadora permitiria:

- Reducir el tiempo de diagnostico.
- Apoyar la toma de decisiones tempranas.
- Disminuir perdidas productivas.
- Promover uso mas racional de fungicidas.
- Democratizar acceso a apoyo tecnico mediante tecnologia movil.

En esta fase se investigaran cuales son las clases (enfermedades) que mas impactan a la planta en el pais, permitiendo en las fases posteriores centrarnos en ellas. Para evaluar el exito se utilizaran metricas estandares de clasificacion: Accuracy (exactitud general), Precision por clase (para analizar falsos positivos), Recall por clase (para no omitir enfermedades reales), F1-score y la Matriz de confusion.

De forma tecnica se tiene como restricciones el uso prioritario en dispositivos Android de gama media o baja, la posibilidad de uso en zonas con conectividad intermitente y la dependencia de disponibilidad de imagenes reales.

### Comprension de los datos

En esta fase se analizaran las fuentes de datos disponibles, se realizara una consolidacion multi-fuente y analisis exploratorio, priorizando datasets con fotografias de campo y fotos en entornos "de estudio" tratadas mediante tecnicas como data augmentation (revisar listado inicial en Recopilacion de datasets).

Han sido considerados como fuentes repositorios academicos como Kaggle, Zenodo, Mendeley Data y repositorios institucionales, registrando y analizando por dataset:

- Licencia y permisos de uso.
- Tipo de imagen (laboratorio vs campo).
- Etiquetas (definicion de clase).
- Distribucion por clase y posibles sesgos.

Se priorizaran imagenes en entornos reales, pero se contempla emplear datasets de imagenes en entornos de estudio como auxiliares de entrenamiento inicial para robustecer el aprendizaje de patrones generales de enfermedad, siempre y cuando la evaluacion final se realice sobre datos reales.

El alcance del proyecto cubre 9 clases: enfermedades foliares (Roya comun, NCLB, GLS), plaga (Gusano cogollero), hoja sana y deficiencias nutricionales (Nitrogeno, Fosforo, Potasio). Esta clasificacion no afecta la limitante de inferencia en dispositivos moviles dado el uso de arquitecturas livianas (MobileNetV3).

### Preparacion de los datos

Esta fase comprendera:

- Limpieza y eliminacion de imagenes corruptas o mal etiquetadas.
- Estandarizacion del tamano de entrada a 224x224 pixeles.
- Normalizacion de valores de intensidad.
- Division del dataset en entrenamiento, validacion y prueba, garantizando que el conjunto de prueba sea completamente independiente y compuesto prioritariamente por imagenes reales de campo.

Se aplicaran tecnicas de aumento de datos (Data Augmentation) durante el entrenamiento, tales como:

- Rotaciones aleatorias.
- Inversiones horizontales.
- Recortes aleatorios.
- Ajustes moderados de brillo y contraste.

Estas transformaciones buscaran mejorar la robustez del modelo frente a variaciones reales de iluminacion, angulo y condiciones ambientales. Existen datasets con imagenes ya tratadas, por lo que se analizara su integracion ahorrando tiempo en la preparacion.

### Modelado

Se empleara una estrategia de transferencia de aprendizaje utilizando la arquitectura MobileNetV3, aunque se consideraran y compararan otras arquitecturas como MobileNetV2 y EfficientNet (B0-B3). MobileNetV3 se priorizara por su bajo numero de parametros, menor consumo computacional y compatibilidad con entornos moviles. Aqui se llevara a cabo un proceso de tres etapas:

1. Inicializacion con pesos preentrenados en ImageNet.
2. Ajuste fino (fine-tuning) con imagenes en condiciones controladas (tratadas mediante data augmentation), para permitir la adaptacion al dominio agricola.
3. Ajuste fino final con imagenes reales de campo para adaptacion de dominio, favoreciendo la adaptacion y reduciendo el sesgo hacia imagenes controladas.

El entrenamiento se realizara utilizando el optimizador Adam, debido a su capacidad de convergencia rapida, se definiran hiperparametros como el learning rate inicial, tamano de lotes segun disponibilidad de memoria y la funcion de perdida (categorical cross-entropy).

### Evaluacion

La evaluacion final se realizara sobre un conjunto de prueba independiente no utilizado durante entrenamiento ni validacion. Se reportaran: Macro F1-score (metrica principal), Matriz de confusion, Precision y Recall por clase, Curvas Precision-Recall por clase.

Si es posible, se realizara una comparacion entre las predicciones del modelo y el diagnostico manual realizado por un agricultor con experiencia o un tecnico agricola. Este analisis permitira estimar el nivel de concordancia entre modelo y diagnostico humano. Adicionalmente, se evaluara el tamano final del modelo y la latencia de inferencia en dispositivo fisico de referencia (>=4GB RAM, CPU Snapdragon serie 6xx o equivalente).

El modelo se considerara viable si:

- Mantiene un Macro F1-score competitivo y equilibrado entre clases.
- Presenta tiempos de inferencia adecuados para uso practico en campo.
- Demuestra coherencia razonable con el diagnostico humano.

### Despliegue y distribucion

El modelo final sera convertido a formato TensorFlow Lite (.tflite) aplicando cuantizacion post-entrenamiento a formato Int8, con el objetivo de reducir el tamano del archivo y mejorar la eficiencia de ejecucion en CPU movil.

La funcionalidad principal de la aplicacion permitira la captura de imagen mediante camara, el procesamiento local sin conexion a internet y la visualizacion de clase predicha y nivel de confianza. Se implementara ademas un modulo de sincronizacion que permita:

- Almacenamiento local de imagenes capturadas con etiquetado manual.
- Registro de fecha.
- Registro opcional de ubicacion aproximada (bajo consentimiento).
- Envio seguro de datos a una API cuando exista conexion a internet.

La API permitira la recepcion, almacenamiento y consulta estructurada de imagenes. Este mecanismo permitira la construccion progresiva de un repositorio de imagenes reales capturadas en territorio salvadoreno.

La aplicacion y la API estaran disenas para soportar:

- Consulta de version de modelo disponible.
- Descarga de nuevos modelos TensorFlow Lite cuando exista conectividad.
- Reemplazo seguro del modelo local sin necesidad de reinstalar la aplicacion.

De esta forma, el sistema permitira mejoras iterativas del modelo sin afectar la funcionalidad offline principal. El reentrenamiento no se ejecutara dentro del alcance del proyecto, pero se documentara formalmente el procedimiento tecnico para la curaduria y validacion de nuevas imagenes recolectadas, el balanceo y limpieza del nuevo subconjunto salvadoreno y el fine-tuning incremental del modelo base utilizando las nuevas imagenes. Este protocolo permitira que futuras investigaciones o instituciones academicas puedan ampliar el dataset nacional, mejorar la robustez del modelo y contribuir a la comunidad cientifica y agricola.

## Cronograma

## Recursos necesarios

Para la ejecucion del proyecto, se requieren recursos humanos, tecnologicos, computacionales y operativos que garanticen el adecuado desarrollo, entrenamiento, validacion y despliegue del sistema propuesto.

### Recursos Humanos

- Especialistas en Machine Learning: Responsables del diseno experimental, seleccion de arquitectura, entrenamiento, evaluacion, optimizacion y documentacion tecnica del modelo.
- Desarrolladores Full Stack: Encargados del desarrollo de la API, la aplicacion web progresiva (PWA), la integracion del modelo entrenado y la implementacion del modulo de sincronizacion.

### Recursos Computacionales

El proyecto contempla una estrategia hibrida de entrenamiento, combinando recursos locales y servicios en la nube.

**Entorno local**

Se utilizara un equipo de computo con las siguientes caracteristicas recomendadas:

- Procesador multinucleo moderno.
- 16 GB de memoria RAM (minimo recomendado).
- Unidad de almacenamiento SSD.
- GPU dedicada (preferiblemente NVIDIA compatible con CUDA) para acelerar el entrenamiento.

**Google Colab**

Se empleara Google Colab como entorno de entrenamiento en la nube para:

- Entrenamiento principal del modelo.
- Fine-tuning con transferencia de aprendizaje.
- Experimentacion con diferentes configuraciones.

En caso de ser requerido se contempla la posibilidad de migrar temporalmente a Google Colab Pro o Pro+, cuyo costo mensual es moderado y justificable en fases de optimizacion intensiva.

### Recursos de Software

**Entrenamiento y modelado**

- Python 3.x con TensorFlow, librerias de procesamiento de imagenes (OpenCV, Pillow), herramientas de data augmentation y frameworks para conversion y optimizacion del modelo.

**Backend**

- FastAPI y Base de datos MySQL.

**Frontend (PWA)**

- Vue 3.

**Control de versiones y automatizacion**

- Git y GitHub con posible configuracion de integracion y despliegue continuo (CI/CD).

### Recursos de Datos

- Dataset inicial de imagenes de enfermedades del maiz (fuentes publicas y/o academicas).
- Imagenes reales recopiladas en campo.

### Infraestructura de Despliegue

- Servidor en la nube (Linux).

### Recursos Economicos

- Hosting o VPS.
- Dominio y certificado SSL.
- Eventual suscripcion temporal a Google Colab Pro (si se requiere).
- Costos operativos asociados a recoleccion de imagenes en campo.
- Consumo electrico del equipo local durante entrenamientos prolongados.

## Riesgos

- Existe una alta probabilidad de que el modelo alcance una precision casi perfecta con imagenes de laboratorio de los dataset seleccionados, pero fracase en el campo real debido a variables no controladas como la iluminacion, sombras o maleza de fondo.
- En el contexto agronomico, decirle a un agricultor que su cultivo esta sano cuando en realidad tiene un inicio de una plaga (un falso negativo) es mucho mas destructivo que diagnosticar erroneamente una enfermedad leve en una planta sana (falso positivo).
- Dado que el sistema debe funcionar completamente offline en las parcelas, la inferencia se realizara utilizando el hardware del dispositivo movil del agricultor, los cuales suelen ser telefonos Android de gama baja con memoria RAM y capacidad de procesamiento limitadas. Un modelo pesado drenara la bateria o colapsara la aplicacion.
- Un modelo de vision por computadora es tan bueno como la imagen que procesa. Si el agricultor toma fotografias borrosas, desde muy lejos o a contraluz, el modelo emitira diagnosticos aleatorios.

## Bibliografia

- CRISP-DM Explained: A Proven Data Mining Methodology - Udacity, fecha de acceso: abril 11, 2026, https://www.udacity.com/blog/2025/03/crisp-dm-explained-a-proven-data-mining-methodology.html
- Ministerio de Agricultura y Ganaderia (MAG). (2023). Anuario de Estadisticas Agropecuarias 2022-2023. Direccion General de Economia Agropecuaria, Gobierno de El Salvador, fecha de acceso: abril 11, de 2026, https://www.mag.gob.sv/anuarios-de-estadisticas-agropecuarias/
- El Diario de Hoy. Recuperado de Reporte de cosecha de granos basicos 2023 - el diario de hoy, fecha de acceso: abril 11, de 2026, https://www.elsalvador.com/h-noticias/h-negocios/bcr-agricultura-ganaderia-el-salvador-pib/1135356/2024/
- Deng, J., et al. (2009). ImageNet: A Large-Scale Hierarchical Image Database. https://www.image-net.org/
- TensorFlow. (2024). TensorFlow Lite Guide. https://www.tensorflow.org/lite
- Centro Nacional de Tecnologia Agropecuaria y Forestal (CENTA). (2018). Cultivo de Maiz (Zea mays L.): Guia Tecnica. Ministerio de Agricultura y Ganaderia, El Salvador. fecha de acceso: abril 11, de 2026, https://www.centa.gob.sv/download/guia-tecnica-cultivo-de-maiz/
- Chollet, F. (2021). Deep Learning with Python (2nd ed.). Manning Publications.
