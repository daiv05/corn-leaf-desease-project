# ETAPA 2 — Análisis de brechas de la rúbrica

Fecha de corte: 2026-09-09. Los estados describen evidencia disponible, no intención ni código sin ejecutar.

| Criterio | Puntos | Estado | Evidencia | Qué falta |
|---|---:|---|---|---|
| Optimización e hiperparámetros | 20 | FALTA | El pipeline principal expone hiperparámetros y regularización, pero no se encontró Optuna ni una búsqueda formal. | Diseñar búsqueda solo con train/validación; ejecutar trials; guardar `trials.csv`, mejores parámetros, historia, importancia y comparación contra baseline. |
| Modelos avanzados y ensemble | 20 | PARCIAL | Registro con 8 CNN modernas; 3 modelos principales con resultados documentados; exportación ONNX/TFLite. | Validar artefactos primarios, comparar bajo un protocolo común e implementar soft/weighted voting con pesos aprendidos exclusivamente en validación. |
| Evaluación rigurosa y métricas finales | 15 | PARCIAL | Split estratificado 70/15/15, métricas globales/por clase, matrices, calibración y evaluación por entorno. | Validación cruzada; auditoría de grupos; nuevo holdout final aislado. El test histórico ya fue observado al comparar arquitecturas. |
| Análisis de sesgos y ética | 15 | PARCIAL | EDA de clase, entorno, fuente, resolución, blur y desbalance; métricas por entorno implementadas. | Fairness report reproducible con gaps de F1/recall/precision, soporte vs. rendimiento y límites derivados de metadatos realmente disponibles. |
| Prototipo funcional y desplegado | 15 | PARCIAL | Inferencia CLI, batch, exportación móvil, cuantización, paridad, OOD y contrato React Native documentados. | En este repo no hay API ni UI; falta demostrar una demo operativa, Top-K, latencia real, baja confianza y evidencia de despliegue end-to-end. |
| Impacto social y recomendaciones | 15 | PARCIAL | El README define población objetivo, uso offline y riesgos generales. | Redactar análisis equilibrado, recomendaciones accionables, supuestos medibles, gobernanza y propuestas de política pública sin presentarlas como políticas existentes. |

## Brechas transversales

- No hay `DATASET_ROOT` configurado ni dataset local en este checkout.
- No hay splits, checkpoints ni outputs primarios descargados.
- No existe fingerprint/versionado inmutable del dataset.
- La cifra histórica previa a agosto de 2026 difiere por una imagen entre documentos.
- README y páginas de `docs/es/pipeline/` están desactualizados respecto del código.
- No hay agrupación por planta, sesión, lote o secuencia.
- El test histórico fue usado en comparaciones, por lo que no puede ser el test final ciego de ETAPA 2.
- Las tablas finales todavía no pueden generarse desde CSV/JSON hasta recuperar o producir esos artefactos.

## Próxima puerta de control

No iniciar tuning hasta recuperar/verificar dataset y outputs, definir el protocolo de holdout final y crear pruebas que hagan fallar cualquier intento de usar ese holdout durante selección.
