# Figuras generadas

Las figuras de este directorio forman parte de la entrega final y se generan a
partir de `outputs/etapa_2/`:

- `dataset_distribution`: distribución por clase y entorno.
- `optuna_history` y `optuna_importance`: búsqueda e importancia relativa.
- `model_comparison` y `ensemble_comparison`: comparación en validación.
- `cross_validation`: macro-F1 de los cinco folds.
- `final_confusion_matrices` y `final_per_class`: holdout abierto una vez.
- `fairness_environment`: composición y rendimiento laboratorio/campo.
- `final_gradcam_cases`: cinco casos reales con Grad-CAM ponderado.
- `final_lime_case`: explicación local del error de mayor confianza.
- `prototype_capture`: entrada real y salida reproducible de la demo.

Los activos estadísticos se regeneran con:

```bash
python scripts/etapa_2/generate_report_assets.py
```

Los dos activos de interpretabilidad se reconstruyen antes con
`scripts/etapa_2/generate_interpretability.py`. El archivo
`../MANIFEST.generated.json` conserva sus rutas y hashes. PDF se usa en gráficas
vectoriales y PNG en imágenes originales, mapas explicativos y captura de demo.
