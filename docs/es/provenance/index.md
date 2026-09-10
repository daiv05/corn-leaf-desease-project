# Procedencia y fuga: plan de experimentación

Plan de trabajo para determinar **hasta dónde se puede resolver** el sesgo de procedencia
del corpus y **dónde conviene detenerse y documentarlo** como limitación del modelo.

- **Horizonte:** 3–6 semanas.
- **Entregable que manda:** tesis y defensa académica.
- **Restricción de datos:** no existe un dataset público de campo para roya común. Sólo
  imágenes sueltas de búsquedas, sin procedencia ni licencia trazables.

## Punto de partida

Un modelo `efficientnet_lite0` entrenado sobre la partición actual alcanza macro-F1 0,9468
en test. Un brazo que ve **únicamente el anillo exterior del 10 % de la imagen** —sin hoja,
sin lesión— alcanza 0,7073, el 84,1 % de ese rendimiento, con 78,3 % de acierto frente a un
azar de 11,1 %. El detalle está en [Fase 0](/es/provenance/fase-0-auditoria).

La consecuencia es que las métricas actuales no distinguen «diagnostica bien» de «reconoce
la sesión fotográfica», y por tanto **0,9468 no es una línea base honesta**.

## Precedentes

El fenómeno está documentado desde 2011 y es la razón por la que la imagen médica adoptó
particiones agrupadas por sitio.

| Trabajo | Hallazgo |
|---|---|
| Torralba y Efros, CVPR 2011 | Un SVM sobre descriptores de bajo nivel predice de qué dataset viene una imagen, muy por encima del azar |
| Zech et al., PLOS Medicine 2018 | CNN de neumonía identifican el hospital de origen en el 99,95 % de las radiografías |
| Bissoto et al., CVPRW 2019 | Borrando la lesión y dejando sólo la piel, la red supera el AUC de los dermatólogos |
| DeGrave et al., Nature MI 2021 | Los detectores de COVID en radiografía eligen atajos antes que señal |
| Noyan, arXiv 2022 | En PlantVillage, 8 píxeles de fondo dan 49,0 % de acierto frente a un azar de 2,6 % |

Sobre qué mitigaciones rinden, dos resultados acotan el esfuerzo razonable:

- **DomainBed** (Gulrajani y Lopez-Paz, ICLR 2021): ningún algoritmo de generalización de
  dominio supera a ERM bien ajustado por más de un punto.
- **Idrissi et al., CLeaR 2022**: el balanceo simple de clases y grupos iguala al estado del
  arte en precisión del peor grupo, y la información de grupo es más crítica para
  *seleccionar* el modelo que para entrenarlo.

La conclusión operativa es que esto se corrige con datos y particiones, no con arquitectura
ni con funciones de pérdida.

## Fases y compuertas

Cada fase cierra con una compuerta de decisión de criterio fijado **antes** de ejecutarla.
El plan está ordenado para que detenerse al final de cualquier fase deje un resultado
defendible.

### Fase 0 — Instrumentar · 2–3 días

Derivar `source_id` por imagen, verificarlo contra la lista de datasets documentada, y medir
los mecanismos de fuga presentes en la partición actual.

**Estado: completada.** Resultados en [Fase 0](/es/provenance/fase-0-auditoria).

> **Compuerta 0.** Una clase con una sola fuente efectiva no admite evaluación honesta y se
> documenta como tal. Resultado: el mínimo observado son 2 fuentes (`lethal_necrosis`), así
> que ninguna clase queda excluida por este criterio.

### Fase 1 — La partición honesta · semana 1–2

Sustituir el splitter estratificado por agrupación sobre `source_id`, y montar validación
**dejando una fuente fuera** (14 pliegues) como evaluación de referencia. Reentrenar con
tres semillas.

Se espera una caída fuerte respecto a 0,9468. **La caída es el resultado, no una regresión.**

> **Compuerta 1 — decisión principal.** Con el F1 por clase bajo validación dejando una
> fuente fuera:
>
> | Criterio | Clasificación | Consecuencia |
> |---|---|---|
> | F1 ≥ 0,70 y el anillo recupera < 60 % | Sostenida | Entra en el sistema |
> | 0,40 ≤ F1 < 0,70 | Frágil | Sólo con predicción selectiva |
> | F1 < 0,40 o el anillo recupera ≥ 80 % | No soportada | Se documenta y sale del alcance |

### Fase 2 — Cuánto se recupera con datos · semana 2–3

Sólo sobre las clases frágiles, y en este orden porque es el que la literatura respalda:

1. **Balanceo de grupos**: submuestrear las celdas fuente×clase sobredimensionadas.
2. **BackMix**: sustituir el fondo en vez de eliminarlo, aprovechando que las imágenes
   pre-enmascaradas de roya común aportan su máscara sin coste.

> **Compuerta 2.** Si una clase frágil no gana al menos 0,10 de F1 con ninguna de las dos
> intervenciones, se declara **techo alcanzado con los datos disponibles** y pasa a
> limitación documentada. No hay tercera intentona.

### Fase 3 — Predicción selectiva · semana 3–4

Recalibrar el detector OOD por distancia de Mahalanobis relativa **contra el eje de fuente**,
no sólo contra clases desconocidas, y reportar la curva riesgo–cobertura.

El entregable deja de ser «un clasificador mejor» y pasa a ser «un clasificador que sabe
cuándo no responder».

### Fase 4 — Redacción · semana 4–6

Limitaciones medidas, no estimadas, y el protocolo de fuga como aportación metodológica
reutilizable.

## Límites asumidos por adelantado

Tres cosas no se resuelven en este horizonte. Declararlo ahora evita gastar semanas en
descubrirlo.

**Roya común en campo.** 106 imágenes reales frente a 2 150 de laboratorio, sin dataset
público disponible. Las imágenes sueltas de búsquedas no son una fuente: sin procedencia ni
licencia trazables agravan justamente el problema que se está midiendo.

**`lethal_necrosis` con dos fuentes.** Admite dos pliegues, pero con intervalo de confianza
ancho. Se reporta con esa salvedad.

**Heterogeneidad de encuadre entre clases.** Que unas clases sean primer plano y otras fotos
de planta entera es una propiedad de la procedencia. No se corrige con preprocesado: es lo
que intentó la segmentación y empeoró el resultado.

## Fuera de alcance

- **IRM, GroupDRO y eliminación adversaria de dominio.** DomainBed acota la ganancia
  esperable a menos de un punto sobre ERM bien ajustado.
- **Reintroducir la segmentación.** Medida y cerrada.
- **Cambiar de backbone buscando número.** Un modelo con más capacidad aprende mejor el
  atajo; no diagnostica mejor.
- **Reportar 0,9468 como resultado.** Sólo como contraste explícito frente a la partición
  honesta.

## Referencias

- [Unbiased Look at Dataset Bias (CVPR 2011)](https://people.csail.mit.edu/torralba/publications/datasets_cvpr11.pdf)
- [Variable generalization performance of a deep learning model to detect pneumonia (PLOS Medicine 2018)](https://journals.plos.org/plosmedicine/article?id=10.1371%2Fjournal.pmed.1002683)
- [(De)Constructing Bias on Skin Lesion Datasets (CVPRW 2019)](https://arxiv.org/abs/1904.08818)
- [AI for radiographic COVID-19 detection selects shortcuts over signal (Nature MI 2021)](https://www.nature.com/articles/s42256-021-00338-7)
- [Uncovering bias in the PlantVillage dataset (2022)](https://arxiv.org/abs/2206.04374)
- [Shortcut Learning in Deep Neural Networks (Nature MI 2020)](https://www.nature.com/articles/s42256-020-00257-z)
- [In Search of Lost Domain Generalization (ICLR 2021)](https://arxiv.org/pdf/2007.01434)
- [Simple data balancing achieves competitive worst-group-accuracy (CLeaR 2022)](https://arxiv.org/abs/2110.14503)
- [BackMix (TPAMI 2025)](https://arxiv.org/abs/2503.17717)
- [CLAIM: Checklist for AI in Medical Imaging, actualización 2024](https://pubs.rsna.org/doi/full/10.1148/ryai.240300)
