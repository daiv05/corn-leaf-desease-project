# DoctorMaiz — Detección de Enfermedades Foliares en Cultivos de Maíz

> **Clasificación mediante Deep Learning en Dispositivos Móviles (Edge AI Offline)**

Proyecto académico de la **Universidad de El Salvador** — Facultad de Ingeniería y Arquitectura  
Especialización en Machine Learning · Ciclo I 2026 · Grupo 02  
Docente: Ing. Bladimir Díaz Campos

---

## Descripción

DoctorMaiz es un sistema de clasificación de enfermedades foliares en cultivos de maíz orientado a pequeños agricultores de subsistencia en zonas rurales sin conectividad. Utiliza un modelo de Deep Learning cuantizado (TensorFlow Lite Int8) embebido en una aplicación Android que opera completamente offline.

### Problema

- El maíz representa una fuente crítica de alimentación en El Salvador, donde la agricultura aporta el **5.6% del PIB**
- El **82.1% de los productores** son pequeños agricultores con acceso limitado a asistencia técnica
- Las enfermedades foliares pueden destruir hasta el **70% de una cosecha**
- El diagnóstico actual depende de experiencia empírica y no de análisis técnico objetivo

### Solución

Una aplicación móvil que, dada una fotografía de hoja de maíz, identifica la enfermedad presente y orienta al agricultor sobre el tratamiento adecuado — sin necesidad de conexión a internet.

---

## Clases Objetivo

| Clase | Patógeno | Síntomas | Imágenes únicas | Campo real |
|---|---|---|---|---|
| Roya común *(Common Rust)* | *Puccinia sorghi* | Pústulas anaranjadas en ambas caras | ~1 591 | ~399 ⚠️ |
| Tizón foliar del norte *(NCLB)* | *Exserohilum turcicum* | Lesiones alargadas grisáceas | ~6 760 | ~5 775 |
| Mancha gris *(GLS)* | *Cercospora zeae-maydis* | Lesiones rectangulares grises | ~5 950 | ~5 437 |
| Hoja sana *(Healthy)* | — | Sin síntomas visibles | ~4 828 | ~3 666 |

> **Nota crítica**: Roya común cuenta con apenas ~399 imágenes de campo real, 14× menos que NCLB. Esta clase requiere data augmentation prioritaria.

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

Se consolidaron **5 fuentes de datos públicas** para construir el corpus de entrenamiento:

| Dataset | Dominio | Imágenes (maíz) | Licencia |
|---|---|---|---|
| [Maize in Field](docs/es/datasets/maize-in-field-dataset.md) | Campo real (Sudáfrica) | ~2 223 | CC BY-NC-SA 4.0 |
| [Maize Diseases](docs/es/datasets/maize-diseases.md) | Lab + campo (PlantVillage v1.0/v1.1) | ~16 162 | CC BY-NC-SA 4.0 |
| [Corn Leaf Diseases](docs/es/datasets/corn-leaf-diseases.md) | Lab augmentado (×17) | 52 360 | MIT |
| [CropDG Unified Multidomain](docs/es/datasets/cropdg-unified-multidomain.md) | Multi-dominio | ~13 275 | CC BY-NC-SA 4.0 |
| [Maize, Beans & Tomatoes Africa](docs/es/datasets/maize-beans-tomatoes-africa.md) | Campo real (África) | 23 286 | Apache 2.0 + CC |

### Inventario Consolidado (Originales Únicos)

| Clase | Lab original | Campo real | Total |
|---|---|---|---|
| Roya común | ~1 192 | ~399 | **~1 591** ⚠️ |
| NCLB | ~1 177 | ~5 775 | **~6 760** |
| GLS | ~581 | ~5 437 | **~5 950** |
| Sano | ~2 203 | ~3 666 | **~4 828** |

### Estrategia de Augmentation

El dataset *Corn Leaf Diseases* aplica 17 técnicas de augmentation documentadas:

`brightness_adjusted` · `contrast_adjusted` · `cropped` · `flipped_horizontal` · `flipped_vertical` · `gaussian_noise` · `high_pass` · `hist_equalized` · `jittered` · `laplacian` · `poisson_noise` · `rotated` · `salt_pepper_noise` · `saturation_adjusted` · `sobel` · `translated` · `unsharp_mask`

Se aplicará un ratio equivalente (×17) a las 399 imágenes de campo de Roya común para obtener ~6 783 imágenes adicionales.

---

## Metodología

El proyecto sigue el marco **CRISP-DM iterativo**:

1. **Comprensión del negocio** — Definición del problema agrícola y restricciones de despliegue
2. **Comprensión de datos** — Consolidación y auditoría de 5 fuentes públicas
3. **Preparación** — Limpieza, estandarización (224×224 px), deduplicación, augmentation
4. **Modelado** — Fine-tuning de MobileNetV3 (baseline V2 y EfficientNet-B0)
5. **Evaluación** — Macro F1 ≥ 0.85 en conjunto independiente de imágenes de campo real
6. **Despliegue** — PWA offline con TFLite Int8 + sincronización opcional

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

## Stack Tecnológico

| Capa | Tecnología | Versión |
|---|---|---|
| Documentación | VitePress | 2.0.0-alpha.17 |
| Framework UI | Vue 3 | (incluido en VitePress) |
| Build tool | Vite | ^8.0.0 |
| Lenguaje | TypeScript | ^5.6.3 |
| Matemáticas | markdown-it-mathjax3 | ^4.3.2 |
| Minificación | oxc-minify (Rust) | ^0.132.0 |

### Componentes personalizados

- **`HeroLogo.vue`** — Logo con efecto glassmorphism y gradiente verde/amarillo
- **`ImageCarousel.vue`** — Carrusel con zoom 1×–3×, descarga y panel de miniaturas
- **`HomeLayout.vue`** — Layout de página de inicio con branding del proyecto

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

- [x] Documentación de 5 datasets consolidados
- [x] Página de inicio con branding profesional
- [x] Componentes Vue interactivos (carrusel, hero logo)
- [x] Sistema de temas con glassmorphism (claro/oscuro)
- [x] Soporte de ecuaciones LaTeX vía MathJax
- [ ] Análisis exploratorio de datos (EDA)
- [ ] Pipeline de preparación de datos
- [ ] Entrenamiento y evaluación del modelo
- [ ] Aplicación Android con TensorFlow Lite

---

## Licencia

Este proyecto es de carácter académico. Los datasets utilizados mantienen sus licencias originales (ver documentación individual de cada dataset).
