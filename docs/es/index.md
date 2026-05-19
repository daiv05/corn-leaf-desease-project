---
layout: home

hero:
  name: "DoctorMaiz"
  text: "Enfermedades Foliares en Maíz"
  tagline: "Sistema de clasificación de enfermedades para pequeños productores de maíz en El Salvador, con inferencia completamente offline en dispositivos Android de gama media/baja."
  actions:
    - theme: brand
      text: Ver Datasets
      link: /es/datasets/

features:
  - icon: "🌽"
    title: "4 Clases Foliares"
    details: "Clasificación de roya común, tizón foliar del norte, mancha gris de la hoja y hoja sana mediante CNN con transferencia de aprendizaje."
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
| **Institución** | Universidad de El Salvador — Facultad de Ingeniería y Arquitectura |
| **Programa** | Escuela de Ingeniería de Sistemas Informáticos — Curso de Especialización en Machine Learning |
| **Ciclo** | I 2026 — Grupo 02 |
| **Docente** | Ing. Bladimir Díaz Campos |

### Equipo

| Nombre | Carnet |
|---|---|
| Josias Abner Rivas Fuentes | RF20010 |
| David Alejandro Deras Cerros | DC19019 |
| Elmer Edenilson Rosales Molina | RM20001 |

### Contexto y Problema

La agricultura representa el 5.6% del PIB de El Salvador y es el sustento de más de 2 millones de personas rurales. El 82.1% de los productores son pequeños agricultores, muchos operando a nivel de subsistencia. El maíz —principal cultivo— es vulnerable a enfermedades foliares que, sin detección temprana, pueden destruir hasta el 70% de la cosecha.

En zonas rurales el acceso a asistencia técnica es limitado. Los diagnósticos dependen de la experiencia empírica del agricultor, lo que genera detecciones tardías y pérdidas económicas significativas. En 2023 la cosecha cayó un tercio respecto a 2021.

### Clases Objetivo

| Clase | Patógeno | Síntoma visual |
|---|---|---|
| **Roya común** | *Puccinia sorghi* | Pústulas anaranjadas dispersas en ambas caras de la hoja |
| **Tizón foliar del norte (NCLB)** | *Exserohilum turcicum* | Lesiones alargadas grisáceas o marrones con bordes difusos |
| **Mancha gris de la hoja (GLS)** | *Cercospora zeae-maydis* | Lesiones rectangulares grises o marrones delimitadas por nervaduras |
| **Hoja sana** | — | Sin síntomas foliares de enfermedad |

### Metodología

El proyecto sigue **CRISP-DM** en fases iterativas:

1. **Comprensión del negocio** — análisis del impacto en el sector agrícola salvadoreño
2. **Comprensión de los datos** — consolidación multi-fuente de datasets públicos; ver [Recopilación de datasets](/es/datasets/)
3. **Preparación de los datos** — limpieza, estandarización a 224 × 224 px y data augmentation
4. **Modelado** — fine-tuning de MobileNetV3 preentrenado en ImageNet; comparación con MobileNetV2 y EfficientNet-B0
5. **Evaluación** — Macro F1 ≥ 0.85 sobre conjunto de prueba independiente compuesto por imágenes de campo
6. **Despliegue** — PWA con inferencia TFLite offline + módulo opcional de sincronización

### Arquitectura del Sistema

```
Captura / Galería
      ↓
Preprocesamiento (224×224, normalización)
      ↓
Inferencia CNN — TensorFlow Lite (Int8)
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
