# DoctorMaiz - Detección de Enfermedades, Plagas y Deficiencias Nutricionales en Cultivos de Maíz

> **Clasificación mediante Deep Learning en Dispositivos Móviles (Edge AI Offline)**

Proyecto académico de la **Universidad de El Salvador** - Facultad de Ingeniería y Arquitectura  
Especialización en Machine Learning · Ciclo I 2026 · Grupo 02  
Docente: Ing. Bladimir Díaz Campos

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

> Conteos post-limpieza y deduplicación en `data/clean/` (junio 2026). Total consolidado: **25 362 imágenes** (3 551 lab + 21 811 campo real).

> **Nota crítica**: Áfidos (77 imgs), Roya común (solo 106 imgs reales), GLS (1 119 total), Potasio (266), Nitrógeno (523) y Fósforo (612) requieren data augmentation prioritaria. Se está definiendo el techo por clase (500, 1 000 o 2 000 imgs).

---

## Objetivos Técnicos

| Métrica | Meta |
|---|---|
| Macro F1 Score | ≥ 0.85 |
| Tamaño del modelo (post Int8) | ≤ 20 MB |
| Latencia de inferencia | ≤ 300 ms/imagen |
| Dispositivo objetivo | Android ≥ 4 GB RAM, Snapdragon 6xx |
| Arquitectura base | MobileNetV3 (comparando con V2 y EfficientNet-B0) |

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

### Inventario Consolidado post-limpieza (`data/clean/`)

| Clase | Lab | Real | Total | Estado |
|---|---:|---:|---:|---|
| Roya común | 2 150 | 106 | 2 256 | Requiere augmentation real ⚠️ |
| NCLB | 888 | 5 942 | 6 830 | Cubierto ✓ |
| GLS | 513 | 606 | 1 119 | Requiere augmentation ⚠️ |
| Sano | 0 | 8 744 | 8 744 | Cubierto ✓ |
| Cogollero | 0 | 4 858 | 4 858 | Cubierto ✓ |
| Nitrógeno | 0 | 523 | 523 | Requiere augmentation ⚠️ |
| Fósforo | 0 | 612 | 612 | Requiere augmentation ⚠️ |
| Potasio | 0 | 266 | 266 | Requiere augmentation ⚠️ |
| Áfidos | 0 | 77 | 77 | Requiere augmentation ⚠️ |
| **TOTAL** | **3 551** | **21 811** | **25 362** | |

### Estrategia de Augmentation

El dataset *Corn Leaf Diseases* aplica 17 técnicas de augmentation documentadas:

`brightness_adjusted` · `contrast_adjusted` · `cropped` · `flipped_horizontal` · `flipped_vertical` · `gaussian_noise` · `high_pass` · `hist_equalized` · `jittered` · `laplacian` · `poisson_noise` · `rotated` · `salt_pepper_noise` · `saturation_adjusted` · `sobel` · `translated` · `unsharp_mask`

Se aplicará un ratio equivalente (×17) a las imágenes de campo de las clases deficitarias (Roya común, Nitrógeno, Fósforo, Potasio) para alcanzar el umbral de ≥ 2 000 imágenes de campo real por clase.

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
│       └── exploratory-data-analysis/
├── notebooks/
│   └── exploration.ipynb       # Análisis exploratorio
├── public/                     # Activos estáticos
│   ├── logo.svg
│   ├── corn-leaf-desease/      # Imágenes de ejemplo
│   ├── maize-diseases/
│   └── maize-in-field-dataset/
├── package.json
└── tsconfig.json
```

---

## Estado del Proyecto

- [x] Documentación de datasets consolidados (8 fuentes, 9 clases)
- [x] Scripts de limpieza y organización de datos en `data/clean/`
- [ ] Análisis exploratorio de datos (EDA)
- [ ] Data augmentation para clases deficitarias (Roya, N, P, K)
- [ ] Pipeline de preparación de datos
- [ ] Entrenamiento y evaluación del modelo
- [ ] Aplicación Android con TensorFlow Lite

---

## Licencia

Este proyecto es de carácter académico. Los datasets utilizados mantienen sus licencias originales (ver documentación individual de cada dataset).
