# Manifiesto del informe LaTeX

Este manifiesto hace localizables las piezas del informe desde Codebase Memory
y separa fuentes editables de activos generados.

## Documento raíz

- `main.tex`: formato A4, portada UES, índices, orden modular y bibliografía.
- `main.pdf`: entrega compilada y revisada, 36 páginas.
- `referencias.bib`: nueve fuentes citadas en el texto.
- `MANIFEST.generated.json`: hashes y rutas de tablas y figuras generadas.

## Secciones incluidas

- `sections/01_resumen.tex`: síntesis, cifras finales y alcance.
- `sections/02_introduccion.tex`: problemas heredados y alcance de Etapa 2.
- `sections/03_objetivos.tex`: objetivo general y nueve objetivos específicos.
- `sections/04_dataset.tex`: 33 433 imágenes, fuentes, duplicados y particiones.
- `sections/07_optimizacion.tex`: baseline, 30 trials Optuna y mejor configuración.
- `sections/08_modelos_avanzados.tex`: cuatro backbones bajo protocolo común.
- `sections/09_ensemble.tex`: complementariedad, voto uniforme y ponderado.
- `sections/10_validacion_cruzada.tex`: cinco folds del conjunto de desarrollo.
- `sections/11_evaluacion_final.tex`: apertura única del holdout y métricas.
- `sections/12_analisis_errores.tex`: N/P/K, roya/tizón y confianza alta.
- `sections/15_interpretabilidad.tex`: Grad-CAM ponderado y LIME local.
- `sections/13_fairness.tex`: clase, entorno, fuente, nitidez y resolución.
- `sections/14_etica.tex`: comunicación, costos de error y trazabilidad.
- `sections/16_prototipo.tex`: app inspeccionada, demo y brecha de despliegue.
- `sections/17_impacto_social.tex`: beneficios posibles y riesgos verificables.
- `sections/18_politica_publica.tex`: nueve recomendaciones concretas.
- `sections/19_limitaciones.tex`: doce límites reales de datos y método.
- `sections/21_conclusiones.tex`: hallazgos y calificación 96/100.

## Tablas generadas

`tables/` contiene: comparación de dataset, configuración y trials de tuning,
baseline frente a tuning, modelos, ensembles, validación cruzada, métricas
finales, resultados por clase, comparación entre etapas, gaps de fairness y
rúbrica. `scripts/etapa_2/generate_report_assets.py` las reconstruye desde
`outputs/etapa_2/` y `docs/etapa_2/rubrica_final.json`.

## Figuras generadas

`figures/` contiene: distribución del dataset, historial e importancia de
Optuna, comparación de modelos y ensembles, variación entre folds, matrices de
confusión, métricas por clase, comparación de entornos, cinco casos Grad-CAM,
un caso LIME y la salida real de la demo. La interpretabilidad procede del
holdout final; no se reciclaron mapas históricos como evidencia nueva.

## Evidencia complementaria

- `docs/etapa_2/11_trazabilidad_rubrica.md`: criterio, puntaje y ruta.
- `docs/etapa_2/rubrica_final.json`: suma auditable de 96/100.
- `docs/etapa_2/02_prototipo_verificado.md`: inspección y pruebas de la app.
- `scripts/etapa_2/stage2_experiments.py`: protocolo experimental completo.
- `tests/etapa_2/test_stage2_experiments.py`: pruebas unitarias y portabilidad
  del clasificador numérico.

El PDF no afirma ajuste fino end-to-end, benchmark Android ni disponibilidad
pública continua: esas tres capacidades no fueron demostradas en la corrida.
