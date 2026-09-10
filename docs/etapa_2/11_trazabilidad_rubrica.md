# Trazabilidad de la rúbrica — ETAPA 2

Corte final: 9 de septiembre de 2026. Los puntos se asignan solo cuando existe
código ejecutado y un CSV/JSON, figura, prueba o sección que permite revisar el
resultado. El archivo estructurado equivalente es
`docs/etapa_2/rubrica_final.json`.

| Criterio | Máximo | Otorgado | Estado | Evidencia principal |
|---|---:|---:|---|---|
| Optimización e hiperparámetros | 20 | 20 | Completo | `outputs/etapa_2/tuning/summary.json`, `trials.csv`, `parameter_importance.csv`, `scripts/etapa_2/stage2_experiments.py`, sección 4 |
| Modelos avanzados y ensemble | 20 | 20 | Completo | `model_comparison/models.csv`, probabilidades por modelo, `ensemble/complementarity.csv`, `ensemble/selection.json`, secciones 5–6 |
| Evaluación rigurosa | 15 | 15 | Completo | `holdout.lock.json`, `cross_validation/folds.csv`, `final/metrics.json` con `evaluation_count=1`, matriz y reporte, secciones 7–9 |
| Sesgos y ética | 15 | 15 | Completo | `fairness/gaps.csv`, `by_group.csv`, `environment_within_class.csv`, `metadata_limitations.json`, secciones 10–12 |
| Prototipo | 15 | 11 | Parcial | demo real, app móvil en commit verificado, 276 pruebas y typecheck; API sin DNS y APK HTTP 404 al cierre, sección 13 |
| Impacto y recomendaciones | 15 | 15 | Completo | condiciones de uso, diseño de evaluación de impacto, nueve propuestas, secciones 14–16 |
| **TOTAL** | **100** | **96** | **Con una salvedad** | Cuatro puntos no otorgados por disponibilidad pública y falta de integración Android del modelo experimental final |

## Evidencia cuantitativa

### Dataset y prevención de fugas

- Archivos verificados: 33.437.
- Duplicados SHA-256 retirados antes del split: 4.
- Imágenes únicas: 33.433.
- División: 23.403 entrenamiento, 5.015 validación y 5.015 holdout.
- Fingerprint:
  `c44de0783d3c3f88ef6151eaa75536d1020f5b208327419c4cb7f3ac459d3892`.
- Evidencia: `outputs/etapa_2/master_manifest.csv`,
  `dataset_summary.json`, `exact_duplicates_removed.csv` y
  `holdout.lock.json`.

### Optimización — 20/20

- 30 trials solicitados, 17 completos y 13 podados.
- Baseline interno EfficientNet-B0: macro-F1 0,7799.
- Mejor trial: macro-F1 0,8887; ganancia absoluta 0,1088.
- Configuración: L2, alpha 4,2836e-6, eta0 0,0006059, batch 512,
  balance_power 0,5, sin average y sin normalización L2.
- El baseline histórico 0,9146 no se usa como comparación controlada porque
  corresponde a otro corpus y fine-tuning end-to-end.

### Modelos y ensemble — 20/20

- EfficientNet-B0: macro-F1 0,8887.
- ShuffleNetV2-x1.0: 0,8694.
- MobileNetV3 Small: 0,8651.
- FastViT-T8: 0,8593.
- Mejor individuo: EfficientNet-B0.
- Soft voting uniforme: 0,9116.
- Weighted soft voting: 0,9202; mejora 0,0315 frente al mejor individuo.
- Los pesos se ajustaron únicamente en validación.

### Evaluación — 15/15

- Cinco folds: macro-F1 medio 0,9097; desviación estándar 0,0090.
- Holdout abierto una vez: accuracy 0,9537; precision macro 0,9012; recall
  macro 0,9092; macro-F1 0,9035; F1 ponderado 0,9537.
- El JSON final conserva el hash del manifiesto del holdout y
  `evaluation_count=1`.

### Sesgos y ética — 15/15

- Gap F1 por clase: 0,3206.
- Gap F1 por fuente: 0,3151.
- Gap F1 por blur: 0,0929.
- Gap F1 por entorno: 0,0861, con composición de clases distinta.
- Gap F1 por resolución: 0,0683.
- Roya: recall 1,0000 en 322 imágenes de laboratorio y 0,3125 en 16 de campo.
  El soporte de campo es pequeño y se reporta junto al valor.
- La sección ética conecta falsos negativos, insumos, OOD, confianza,
  ubicación opcional y trazabilidad con controles reales de la app.

### Prototipo — 11/15

- Demo de la selección final: `outputs/etapa_2/prototype/demo_result.json`.
- App React Native/Expo con EfficientNet-Lite0 TFLite int8, inferencia local,
  OOD y base offline.
- Artefacto móvil: 3.743.824 bytes; SHA-256 documentado en
  `docs/etapa_2/02_prototipo_verificado.md`.
- `npm run typecheck`: aprobado.
- `npm test -- --runInBand`: 40 suites y 276 pruebas aprobadas.
- No se otorgan cuatro puntos: la aplicación experimental final no se exportó
  e integró en Android; el dominio no resolvió y el APK no fue público durante
  la verificación final.

### Impacto y recomendaciones — 15/15

- Se distingue capacidad técnica de impacto medido.
- Se proponen condiciones de validación prospectiva con productores y
  extensionistas.
- Se incluyen nueve medidas: dataset nacional, captura, revisión agronómica,
  extensión, monitoreo, ubicación, ciclo de feedback, validación de campo y
  ficha versionada.

## Reproducibilidad

El código principal está en `scripts/etapa_2/stage2_experiments.py`.
Las figuras y tablas se regeneran con
`scripts/etapa_2/generate_report_assets.py`; las explicaciones, con
`generate_interpretability.py`; la consolidación numérica, con
`summarize_results.py`. Las pruebas del protocolo están en
`tests/etapa_2/test_stage2_experiments.py`.

Los outputs voluminosos permanecen ignorados por Git. El informe versiona las
tablas, figuras, manifiesto de generación y PDF necesarios para auditar las
cifras que aparecen en la entrega.
