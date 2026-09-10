# Auditoría inicial del clasificador

Fecha de corte: 2026-09-09. Commit auditado: `7314344d1bb53a3ca5444e46f24303b9076fea65`.

Esta auditoría se realizó antes de iniciar entrenamientos de ETAPA 2. Distingue el código disponible, la evidencia documental histórica y los artefactos experimentales realmente presentes en el checkout.

## Codebase Memory

- Disponible: **sí**, mediante MCP `codebase-memory-mcp`.
- Configuración previa del repositorio: no existía `.codex/config.toml`.
- Consulta inicial: el repositorio no estaba indexado; el servidor mostraba otros cuatro proyectos.
- Acción: indexación completa ejecutada el 2026-09-09 a las 08:57 CST.
- Resultado: proyecto `home-desarrolloab-Documentos-ML-Clasificador-maize-doctor-classifier`, 1 999 nodos y 7 549 relaciones.
- Configuración añadida: caché persistente dedicada en `/home/desarrolloab/Documentos/CodebaseMemory/maize-doctor-classifier` para sesiones futuras.
- Necesita reindexación: **sí después de los cambios documentales de esta auditoría**; se hará al cerrar esta fase.

## Estado material del checkout

- No existe `.env`; por tanto `DATASET_ROOT` no está configurado en esta copia.
- No existen `outputs/`, `outputs-remote/` ni `checkpoints/`.
- Sí existen código de entrenamiento, evaluación, inferencia, exportación y explicabilidad; una suite de pruebas; documentación VitePress; EDA; figuras; y el informe congelado de la primera etapa.
- Las métricas históricas pueden citarse con su procedencia documental, pero no se consideran reproducidas en esta máquina mientras falten sus CSV/JSON, splits y checkpoints.

## Dataset actual documentado

`config/dataset.yaml` define 9 clases, tamaño objetivo 224×224, semilla 42 y umbral de clase minoritaria 4,0. La documentación posterior a la ampliación de agosto de 2026 informa 33 438 imágenes: 3 551 de laboratorio y 29 887 de campo.

| Clase | Lab | Real | Total |
|---|---:|---:|---:|
| `common_rust` | 2 150 | 106 | 2 256 |
| `fall_armyworm` | 0 | 4 858 | 4 858 |
| `gray_leaf_spot` | 513 | 1 417 | 1 930 |
| `healthy` | 0 | 8 744 | 8 744 |
| `lethal_necrosis` | 0 | 6 415 | 6 415 |
| `nitrogen_deficiency` | 0 | 846 | 846 |
| `northern_corn_leaf_blight` | 888 | 5 942 | 6 830 |
| `phosphorus_deficiency` | 0 | 938 | 938 |
| `potassium_deficiency` | 0 | 621 | 621 |
| **Total** | **3 551** | **29 887** | **33 438** |

El dataset no tiene en el repositorio un identificador inmutable de versión o fingerprint. Se lo identifica solo como “ampliación de agosto de 2026”. Hay una inconsistencia documental que debe resolverse: algunos documentos indican 31 622 imágenes antes de la ampliación, mientras `CLAUDE.md` indica 31 623; el total actual y la adición neta documentada de 1 815 son aritméticamente compatibles con 31 623.

## Preparación y splits

El pipeline `scripts/pipeline/create_splits.py`:

- valida cada imagen con PIL;
- calcula SHA-256 y omite duplicados exactos antes de dividir;
- recorre rutas en orden estable;
- estratifica por combinación `label + environment`;
- usa una partición 70/15/15 con semilla 42;
- escribe `train.csv`, `val.csv`, `test.csv` y `split_audit_report.csv`.

La documentación también registra una deduplicación perceptual previa con pHash. No obstante, el split actual es por imagen: no hay identificadores de planta, sesión, lote, secuencia o grupo, ni `StratifiedGroupKFold`. Por ello no puede descartarse fuga por imágenes correlacionadas o variantes cercanas entre particiones.

## Baseline oficial histórico

El baseline oficial documentado es la comparación de `efficientnet_b0`, `shufflenet_v2_x1_0` y `efficientnet_lite0` con pesos ImageNet, 30 épocas, AdamW (`lr=1e-4`, `weight_decay=1e-4`), batch 32, `CrossEntropyLoss`, `WeightedRandomSampler` y augmentation de clases minoritarias. Usa el perfil baseline de 9 clases con tope de 1 500 imágenes por clase.

La documentación de la primera etapa reporta sobre un test histórico de 1 503 imágenes:

| Modelo | Accuracy | Macro F1 | Loss test | Checkpoint |
|---|---:|---:|---:|---:|
| `efficientnet_b0` | 0,9521 | 0,9146 | 0,229 | 15,6 MB |
| `shufflenet_v2_x1_0` | 0,9508 | 0,9030 | 0,169 | 5,0 MB |
| `efficientnet_lite0` | 0,9474 | 0,8951 | 0,218 | 13,1 MB |

Estas cifras se reutilizan como evidencia histórica, no como evaluación final de ETAPA 2. Los artefactos primarios de esas corridas no están presentes en el checkout.

## Modelos e infraestructura existentes

El registro contiene ocho arquitecturas: EfficientNet-B0, EfficientNet-Lite0, EfficientNet-B4, ShuffleNetV2-x1.0, MobileNetV3 Large, MobileNetV3 Small, FastViT-T8 y GhostNetV2-100. Los tres modelos canónicos tienen evidencia de entrenamiento histórico.

El pipeline principal ya implementa pérdida ponderada (`sqrt_inverse` por defecto), label smoothing 0,1, AdamW, scheduler cosine con 3 épocas de warmup, early stopping con paciencia 8, clipping de gradiente 1,0 y 60 épocas máximas. Esto contradice páginas antiguas del README y de `docs/es/pipeline/`, que aún hablan del pipeline principal en futuro o como pendiente.

Existe infraestructura para:

- métricas globales y por clase, matrices de confusión, predicciones por imagen, calibración y desglose por entorno;
- inferencia por CLI y por lotes;
- LIME, Grad-CAM y SHAP post-hoc;
- exportación ONNX/TFLite, cuantización int8, prueba de paridad y evaluación del artefacto exportado;
- detección fuera de dominio mediante features y distancia de Mahalanobis.

No se encontraron implementaciones de Optuna, validación cruzada, ensemble, informe fairness de ETAPA 2, Streamlit/Gradio, servidor FastAPI/Flask ni rutas HTTP en este repositorio.

## Resultados posteriores documentados

Un plan de despliegue fechado en agosto de 2026 registra, como evidencia secundaria, corridas del pipeline principal sobre un test de 5 015 imágenes y macro-F1 FP32 aproximado de 0,943 (`efficientnet_b0`), 0,947 (`efficientnet_lite0`) y 0,924 (`shufflenet_v2_x1_0`). El mismo documento registra evaluaciones TFLite int8 de 0,9241, 0,9477 y 0,9229 respectivamente.

No se incorporan esas cifras como resultados finales porque los `test_grouped_metrics.json`, `eval_tflite_int8.json`, `summary.json`, `predictions.csv` y checkpoints citados viven en árboles ignorados por Git y no están en esta copia. Deben recuperarse desde el volumen de experimentos y validarse antes de alimentar tablas automáticas de ETAPA 2.

## Riesgo principal: aislamiento del test

Los scripts actuales evalúan `test.csv` al terminar cada corrida y la documentación compara arquitecturas con esas métricas. Eso convierte el test histórico en un benchmark observado durante decisiones de modelado. No hay mezcla automática de filas entre train/val/test, pero sí contaminación del protocolo de selección.

Para ETAPA 2 se debe:

1. conservar el test histórico solo como referencia;
2. definir y versionar un holdout final nuevo antes del tuning;
3. auditar grupos, sesiones y near-duplicates antes de congelarlo;
4. impedir que tuning, early stopping, calibración y pesos del ensemble accedan a ese holdout;
5. evaluarlo una sola vez después de congelar modelo y umbrales.

## Conclusión de la auditoría

El clasificador tiene una base técnica más madura de lo que indica parte de su documentación: el pipeline principal, la XAI y la exportación móvil están implementados y probados a nivel de código. Sin embargo, la rúbrica de ETAPA 2 aún requiere infraestructura experimental formal, recuperación de artefactos, un protocolo final no contaminado, validación cruzada, ensemble, fairness consolidado y evidencia operativa del prototipo.

## Verificación del informe

- `git diff --check`: sin errores de espacios o patch.
- Rutas de las tres figuras iniciales: verificadas.
- Validación estática: 25 archivos `.tex`, sin llaves/entornos desbalanceados ni `\\input` faltantes desde `main.tex`.
- Compilación: ejecutada con el binario portátil oficial Tectonic 0.17.0 después de verificar su SHA-256 publicado.
- Resultado: `informe/main.pdf`, A4, 10 páginas y sin warnings LaTeX, referencias indefinidas ni labels duplicados.
- Instrucciones y dependencias: documentadas en `informe/README.md`.
- Indexabilidad: el extractor activo de Codebase Memory no crea nodos para `.tex`; `informe/MANIFEST.md` enumera todos esos archivos para mantenerlos localizables desde el grafo.
