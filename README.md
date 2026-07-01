# DoctorMaiz - Detección de Enfermedades, Plagas y Deficiencias Nutricionales en Cultivos de Maíz

> **Clasificación mediante Deep Learning en Dispositivos Móviles (Edge AI Offline)**


---

## Descripción

DoctorMaiz es un sistema de clasificación de enfermedades foliares, plagas y deficiencias nutricionales en cultivos de maíz orientado a pequeños agricultores de subsistencia en zonas rurales sin conectividad. Utiliza un modelo de Deep Learning cuantizado (TensorFlow Lite Int8) embebido en una aplicación Android que opera completamente offline.

### Problema

- El maíz representa una fuente crítica de alimentación en El Salvador, donde la agricultura aporta el **5.6% del PIB**
- El **82.1% de los productores** son pequeños agricultores con acceso limitado a asistencia técnica
- Las enfermedades, plagas y deficiencias nutricionales pueden destruir hasta el **70% de una cosecha**
- El diagnóstico actual depende de experiencia empírica y no de análisis técnico objetivo

### Solución

Una aplicación móvil que, dada una fotografía de hoja de maíz, identifica la enfermedad, plaga o deficiencia nutricional presente y orienta al agricultor sobre el tratamiento adecuado - sin necesidad de conexión a internet.

---

## Clases Objetivo

### Enfermedades y plagas foliares

| Clase | Patógeno/Agente | Síntomas | Lab | Real | Total |
|---|---|---|---:|---:|---:|
| Roya común *(Common Rust)* | *Puccinia sorghi* | Pústulas anaranjadas en ambas caras | 2 150 | 106 ⚠️ | 2 256 |
| Tizón foliar del norte *(NCLB)* | *Exserohilum turcicum* | Lesiones alargadas grisáceas | 888 | 5 942 | 6 830 |
| Mancha gris *(GLS)* | *Cercospora zeae-maydis* | Lesiones rectangulares grises | 513 | 606 | 1 119 |
| Hoja sana *(Healthy)* | - | Sin síntomas visibles | 0 | 8 744 | 8 744 |
| Gusano cogollero *(Fall Armyworm)* | *Spodoptera frugiperda* | Daño por masticación, excrementos en cogollo | 0 | 4 858 | 4 858 |
| Áfidos del maíz *(Maize Aphids)* | *Rhopalosiphum maidis* | Colonias de pulgones, hojas enrolladas y amarillamiento | 0 | 77 ⚠️ | 77 |

### Deficiencias nutricionales

| Clase | Síntomas | Lab | Real | Total |
|---|---|---:|---:|---:|
| Deficiencia de nitrógeno *(Nitrogen)* | Amarillamiento en "V" desde puntas de hojas inferiores | 0 | 523 ⚠️ | 523 |
| Deficiencia de fósforo *(Phosphorus)* | Bordes y puntas moradas/rojizas en hojas jóvenes | 0 | 612 ⚠️ | 612 |
| Deficiencia de potasio *(Potassium)* | Necrosis marginal en hojas más viejas | 0 | 266 ⚠️ | 266 |


---

## Objetivos Técnicos

| Métrica | Meta |
|---|---|
| Macro F1 Score | ≥ 0.85 |
| Tamaño del modelo (post Int8) | ≤ 20 MB |
| Latencia de inferencia | ≤ 300 ms/imagen |
| Dispositivo objetivo | Android ≥ 4 GB RAM, Snapdragon 6xx |
| Arquitectura base | -- Por definir -- |

---

## Datasets

Se consolidaron **8 fuentes de datos públicas** para construir el corpus de entrenamiento:

| Dataset | Dominio | Imágenes (maíz) | Licencia |
|---|---|---|---|
| [Maize in Field](docs/es/datasets/maize-in-field-dataset.md) | Campo real (Sudáfrica) | ~2 223 | CC BY-NC-SA 4.0 |
| [Maize Diseases](docs/es/datasets/maize-diseases.md) | Lab + campo (PlantVillage v1.0/v1.1) | ~16 162 | CC BY-NC-SA 4.0 |
| [Corn Leaf Diseases](docs/es/datasets/corn-leaf-diseases.md) | Lab augmentado (×17) | 52 360 | MIT |
| [CropDG Unified Multidomain](docs/es/datasets/cropdg-unified-multidomain.md) | Multi-dominio | ~13 275 | CC BY-NC-SA 4.0 |
| [Maize, Beans & Tomatoes Africa](docs/es/datasets/maize-beans-tomatoes-africa.md) | Campo real (África) | 23 286 | Apache 2.0 + CC |
| [Multicrop Disease - Maize Pests and Disease](docs/es/datasets/multicrop-disease-maiz-disease-pests-and-disease.md) | Mixto | - | Desconocida |
| [Maize Nutrient Deficiency](docs/es/datasets/maize-nutrient-deficiency.md) | Campo real (India) | 463 | CC BY 4.0 |
| [Corn Leaf - Roboflow](docs/es/datasets/corn-leaf-roboflow.md) | Campo real | 3 943 | CC BY 4.0 |


### Estrategia de Augmentation

El dataset *Corn Leaf Diseases* aplica 17 técnicas de augmentation documentadas:

`brightness_adjusted` · `contrast_adjusted` · `cropped` · `flipped_horizontal` · `flipped_vertical` · `gaussian_noise` · `high_pass` · `hist_equalized` · `jittered` · `laplacian` · `poisson_noise` · `rotated` · `salt_pepper_noise` · `saturation_adjusted` · `sobel` · `translated` · `unsharp_mask`

---

## Metodología

El proyecto sigue el marco **CRISP-DM iterativo**:

1. **Comprensión del negocio** - Definición del problema agrícola y restricciones de despliegue
2. **Comprensión de datos** - Consolidación y auditoría de 5 fuentes públicas
3. **Preparación** - Limpieza, estandarización (224×224 px), deduplicación, augmentation
4. **Modelado** - Fine-tuning de MobileNetV3 (baseline V2 y EfficientNet-B0)
5. **Evaluación** - Macro F1 ≥ 0.85 en conjunto independiente de imágenes de campo real
6. **Despliegue** - PWA offline con TFLite Int8 + sincronización opcional

---

## Pipeline de Machine Learning

El código de datos/entrenamiento vive en `src/` (librería instalable) y `scripts/` (entrypoints). Hay
dos pipelines paralelos sobre el mismo dataset limpio (`data/clean/`):

- **Baselines** (`scripts/pipeline/train_baselines.py`): EfficientNet-B0, EfficientNet-Lite0 y MobileNetV3-Large pre-entrenados, funcional de punta a punta. Por defecto entrena sobre un subset configurable (`config/dataset.yaml -> baseline:`, 4 clases y hasta 500 imágenes por clase) para comparar arquitecturas rápido y barato; ver [Baselines](docs/es/baselines/index.md).
- **Pipeline principal** (`scripts/pipeline/train.py`): comparte toda la infraestructura de datos y modelos; el loop de entrenamiento está pendiente de implementar.

### Quickstart

```bash
cp .env.example .env && make install && make download-dataset && make splits-baseline && make train-baselines
```

Guía completa de instalación (venv, `.env`, dataset) en [LOCAL.md](LOCAL.md). Para entrenar en una GPU
alquilada en [vast.ai](https://vast.ai) con el mismo flujo reproducible, ver
[docs/es/deployment/vast-ai.md](docs/es/deployment/vast-ai.md).

---

## Equipo

| Nombre | Carné |
|---|---|
| Josias Abner Rivas Fuentes | RF20010 |
| David Alejandro Deras Cerros | DC19019 |
| Elmer Edenilson Rosales Molina | RM20001 |

---

## Documentación

La documentación completa del proyecto está construida con **VitePress** y se encuentra en el directorio `docs/`.

### Ejecutar la documentación localmente

```bash
# Instalar dependencias
npm install

# Servidor de desarrollo con hot-reload
npm run docs:dev

# Compilar para producción
npm run docs:build

# Vista previa de la compilación
npm run docs:preview

# Verificar tipos TypeScript
npm run typecheck
```

La documentación estará disponible en `http://localhost:5173`.

---

## Estructura del Proyecto

```
corn-leaf-desease-project/
├── config/
│   └── dataset.yaml            # Clases, tamaño de imagen, seed, perfil "baseline"
├── docs/
│   ├── .vitepress/
│   │   ├── components/         # Componentes Vue reutilizables
│   │   │   ├── HeroLogo.vue
│   │   │   └── ImageCarousel.vue
│   │   ├── theme/              # Tema y estilos personalizados
│   │   │   ├── index.ts
│   │   │   ├── HomeLayout.vue
│   │   │   └── custom.css
│   │   └── config.mts          # Configuración VitePress
│   └── es/                     # Documentación en español
│       ├── index.md            # Página de inicio
│       ├── datasets/           # Documentación de datasets
│       ├── exploratory-data-analysis/
│       ├── baselines/          # Baselines de Deep Learning (EfficientNet, MobileNetV3)
│       └── deployment/         # Entrenamiento reproducible en GPU (vast.ai)
├── notebooks/
│   └── 01_eda.ipynb            # Análisis exploratorio
├── public/                     # Activos estáticos
│   ├── logo.svg
│   ├── corn-leaf-desease/      # Imágenes de ejemplo
│   ├── maize-diseases/
│   └── maize-in-field-dataset/
├── scripts/
│   ├── cleanup/                 # Limpieza por clase/dataset (one-shot, ya ejecutados)
│   ├── dataset/                 # Subida/descarga de data/clean/ (Hugging Face Hub, Google Drive)
│   ├── pipeline/                # create_splits.py, train_baselines.py, train.py
│   └── vastai/                  # Orquestación de GPU remota en vast.ai
├── src/                         # Librería principal (pip install -e .)
│   ├── config.py
│   ├── analysis/                # Resumen del dataset
│   ├── cleanup/                 # Deduplicación perceptual (PHash)
│   ├── data/                    # CornDataset, loader, splitter, transforms
│   └── models/                  # Registro de modelos + baselines (EfficientNet, MobileNetV3)
├── Dockerfile                   # Imagen reproducible (Python 3.11 + PyTorch CUDA) para GPU remota
├── pyproject.toml
├── Makefile
├── package.json
└── tsconfig.json
```

---

## Estado del Proyecto

- [x] Documentación de datasets consolidados (8 fuentes, 9 clases)
- [x] Scripts de limpieza y organización de datos en `data/clean/`
- [x] Análisis exploratorio de datos (EDA)
- [x] Pipeline de preparación de datos (splits estratificados, perfil baseline configurable)
- [x] Data augmentation para clases minoritarias (pipeline extendido por clase en `transforms.py`)
- [x] Entrenamiento de baselines (EfficientNet-B0/Lite0, MobileNetV3-Large) + soporte GPU remota (vast.ai)
- [ ] Loop de entrenamiento del pipeline principal (`scripts/pipeline/train.py`)
- [ ] Evaluación exhaustiva y selección de modelo final
- [ ] Aplicación Android con TensorFlow Lite

---

## Licencia

Este proyecto es de carácter académico. Los datasets utilizados mantienen sus licencias originales (ver documentación individual de cada dataset).
