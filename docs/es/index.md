---
layout: home

hero:
  name: "DoctorMaiz"
  text: "Enfermedades Foliares en Maíz"
  tagline: "Sistema de clasificación de enfermedades para pequeños productores de maíz en El Salvador, con inferencia completamente offline en dispositivos Android de gama media/baja."
  image:
    src: /logo.svg
    alt: DoctorMaiz
  actions:
    - theme: brand
      text: Ver Datasets
      link: /es/datasets/

features:
  - icon: "🌽"
    title: "9 Clases Foliares"
    details: "Clasificación de 5 enfermedades/plagas (roya, NCLB, GLS, cogollero, sana) y 3 deficiencias nutricionales (N, P, K) mediante CNN con transferencia de aprendizaje."
  - icon: "📱"
    title: "Edge AI Offline"
    details: "Modelo TensorFlow Lite con cuantización Int8, objetivo ≤ 20 MB y latencia ≤ 300 ms en CPU Snapdragon serie 6xx o equivalente."
  - icon: "🌍"
    title: "Orientado al Campo"
    details: "Evaluación priorizada sobre imágenes reales de campo. El conjunto de prueba es independiente y de dominio real para garantizar robustez agrícola."
  - icon: "📊"
    title: "Meta Macro F1 ≥ 0.85"
    details: "Criterio de viabilidad con análisis de matriz de confusión y curvas Precision-Recall por clase, priorizando Recall para minimizar falsos negativos."
---

## El Proyecto

**Detección de Enfermedades Foliares en Cultivos de Maíz mediante Deep Learning en Dispositivos Móviles**

| | |
|---|---|
| **Institución** | Universidad de El Salvador - Facultad de Ingeniería y Arquitectura |
| **Programa** | Escuela de Ingeniería de Sistemas Informáticos - Curso de Especialización en Machine Learning |
| **Ciclo** | I 2026 - Grupo 02 |
| **Docente** | Ing. Bladimir Díaz Campos |

### Equipo

| Nombre | Carnet |
|---|---|
| Josias Abner Rivas Fuentes | RF20010 |
| David Alejandro Deras Cerros | DC19019 |
| Elmer Edenilson Rosales Molina | RM20001 |

### Contexto y Problema

La agricultura representa el 5.6% del PIB de El Salvador y es el sustento de más de 2 millones de personas rurales. El 82.1% de los productores son pequeños agricultores, muchos operando a nivel de subsistencia. El maíz -principal cultivo- es vulnerable a enfermedades foliares que, sin detección temprana, pueden destruir hasta el 70% de la cosecha.

En zonas rurales el acceso a asistencia técnica es limitado. Los diagnósticos dependen de la experiencia empírica del agricultor, lo que genera detecciones tardías y pérdidas económicas significativas. En 2023 la cosecha cayó un tercio respecto a 2021.

### Clases Objetivo

#### Enfermedades y plagas foliares

| Clase | Nombre en inglés | Patógeno/Agente | Síntoma visual | Campo real |
|---|---|---|---|---|
| **Roya común** | Common Rust | *Puccinia sorghi* | Pústulas anaranjadas dispersas en ambas caras de la hoja | ~399 ⚠️ |
| **Tizón foliar del norte (NCLB)** | Northern Corn Leaf Blight | *Exserohilum turcicum* | Lesiones alargadas grisáceas o marrones con bordes difusos | ~6 067 |
| **Mancha gris de la hoja (GLS)** | Gray Leaf Spot | *Cercospora zeae-maydis* | Lesiones rectangulares grises o marrones delimitadas por nervaduras | ~9 383 |
| **Hoja sana** | Healthy | - | Sin síntomas foliares de enfermedad | ~4 285 |
| **Gusano cogollero** | Fall Armyworm | *Spodoptera frugiperda* | Daño por masticación con excrementos en el cogollo y hojas | ~4 935 |

#### Deficiencias nutricionales

| Clase | Nombre en inglés | Síntoma visual | Campo real |
|---|---|---|---|
| **Deficiencia de nitrógeno** | Nitrogen Deficiency | Amarillamiento en "V" desde la punta de hojas inferiores | ~622 ⚠️ |
| **Deficiencia de fósforo** | Phosphorus Deficiency | Bordes y puntas moradas/rojizas en hojas jóvenes | ~725 ⚠️ |
| **Deficiencia de potasio** | Potassium Deficiency | Necrosis marginal en hojas más viejas | ~322 ⚠️ |

::: warning Desbalance crítico en cuatro clases
**Roya común** (~399 imgs campo), **Potasio** (~322), **Nitrógeno** (~622) y **Fósforo** (~725) no alcanzan el objetivo de ≥ 2 000 imágenes de campo real. Estas clases requieren data augmentation prioritaria antes de la etapa de adaptación de dominio.
:::

### Metodología

El proyecto sigue **CRISP-DM** en fases iterativas:

1. **Comprensión del negocio** - análisis del impacto en el sector agrícola salvadoreño
2. **Comprensión de los datos** - consolidación multi-fuente de datasets públicos; ver [Recopilación de datasets](/es/datasets/)
3. **Preparación de los datos** - limpieza, estandarización a 224 × 224 px y data augmentation
4. **Modelado** - fine-tuning de MobileNetV3 preentrenado en ImageNet; comparación con MobileNetV2 y EfficientNet-B0
5. **Evaluación** - Macro F1 ≥ 0.85 sobre conjunto de prueba independiente compuesto por imágenes de campo
6. **Despliegue** - PWA con inferencia TFLite offline + módulo opcional de sincronización

### Arquitectura del Sistema

```
Captura / Galería
      ↓
Preprocesamiento (224×224, normalización)
      ↓
Inferencia CNN - TensorFlow Lite (Int8)
      ↓
Clase predicha + nivel de confianza
      ↓ (cuando hay conexión)
Módulo de sincronización → API FastAPI + MySQL
```

### Restricciones Técnicas

- Dispositivo objetivo: Android ≥ 4 GB RAM, CPU Snapdragon serie 6xx o equivalente
- Tamaño del modelo: ≤ 20 MB (post cuantización Int8)
- Latencia de inferencia: ≤ 300 ms por imagen
- Funcionamiento completamente offline como característica principal
