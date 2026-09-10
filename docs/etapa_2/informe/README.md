# Informe LaTeX de ETAPA 2

Este directorio contiene el informe final reproducible de la Etapa 2 de Doctor
Maíz. `main.tex` integra 17 secciones numeradas, las tablas generadas desde los
resultados y 13 figuras. El PDF revisado es `main.pdf` (36 páginas A4).

## Compilación

Desde este directorio:

```bash
tectonic main.tex --keep-logs --keep-intermediates
```

También puede usarse una distribución TeX completa:

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

El documento usa español, `natbib`, `microtype`, `booktabs`, `tabularx`,
`longtable`, `float`, `subcaption`, `amsmath`, `fancyhdr` e `hyperref`.

## Fuentes de verdad

- Corrida nueva: `outputs/etapa_2/`.
- Dataset y particiones: `outputs/etapa_2/master_manifest.csv`, `holdout.csv`,
  `holdout.lock.json` y `dataset_summary.json`.
- Selección: `outputs/etapa_2/tuning/`, `model_comparison/`, `ensemble/` y
  `cross_validation/`.
- Única evaluación final: `outputs/etapa_2/final/metrics.json`, con
  `evaluation_count=1`; el comando se niega a reabrir un resultado existente.
- Fairness e interpretabilidad: `outputs/etapa_2/fairness/` e
  `outputs/etapa_2/interpretability/`.
- Evidencia del prototipo: `docs/etapa_2/02_prototipo_verificado.md` y
  `outputs/etapa_2/prototype/demo_result.json`.
- Resultados históricos: `docs/es/pipeline-baselines/` y
  `reports/firts-phase/`; se usan únicamente como referencia orientativa.

## Regeneración de experimentos

Desde la raíz del repositorio, con las dependencias del proyecto instaladas:

```bash
python scripts/etapa_2/stage2_experiments.py prepare --dataset-root /ruta/dataset
python scripts/etapa_2/stage2_experiments.py extract --dataset-root /ruta/dataset
python scripts/etapa_2/stage2_experiments.py tune
python scripts/etapa_2/stage2_experiments.py compare
python scripts/etapa_2/stage2_experiments.py ensemble
python scripts/etapa_2/stage2_experiments.py cv
python scripts/etapa_2/stage2_experiments.py final
python scripts/etapa_2/stage2_experiments.py fairness
python scripts/etapa_2/generate_interpretability.py
python scripts/etapa_2/generate_report_assets.py
```

La extracción de 33 433 imágenes y los cuatro backbones es la parte costosa.
No se debe borrar `outputs/etapa_2/final/` para repetir el holdout como si fuera
una evaluación nueva. Las figuras y tablas del informe se reconstruyen desde
los CSV/JSON ya generados con el último comando.

## Verificación

```bash
pytest tests/etapa_2 -q
pdfinfo docs/etapa_2/informe/main.pdf
pdftotext docs/etapa_2/informe/main.pdf -
```

`MANIFEST.md` enumera el contenido modular. La trazabilidad de cada criterio de
la rúbrica está en `docs/etapa_2/11_trazabilidad_rubrica.md`.
