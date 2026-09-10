# Fase 1 — La partición honesta

Evaluación del corpus con validación **dejando una fuente fuera**, y comparación contra la
partición aleatoria actual. Fecha de ejecución: 2026-09-09. GPU A10 en Modal.

Reproducible con:

```bash
modal run scripts/modal/leave_one_source_out.py
modal volume get corn-outputs experiments/leave_one_source_out.json <destino>
```

Evidencia bruta en [`evidencia/`](/es/provenance/evidencia/): resultado por pliegue,
predicciones por imagen y traza completa de la ejecución.

## Protocolo

Once pliegues, uno por grupo de procedencia. En cada uno:

- **Prueba**: el grupo de procedencia retenido, completo.
- **Validación**: un segundo grupo distinto, rotado, elegido de modo que ninguna clase se
  quede sin ejemplos de entrenamiento. Ni el entrenamiento ni la selección de modelo ven el
  dominio sobre el que se mide.
- **Entrenamiento**: los nueve grupos restantes, con tope de 1 000 imágenes por clase.

Antes de particionar se eliminan casi-duplicados conservando un representante por componente
de hash perceptual idéntico: **165 imágenes descartadas** de 33 433. Sin ese paso la misma
foto podría aparecer a ambos lados de la frontera aunque las fuentes estén separadas.

Cada imagen se evalúa **exactamente una vez**, en el pliegue donde su fuente queda retenida.
Las predicciones agrupadas cubren **33 268 imágenes**, todas fuera de fuente.

El resto del protocolo replica el del brazo `original` del test de fuga —`efficientnet_lite0`
preentrenado, 25 épocas, paciencia 6, lote 64, tasa 1e-4, tope de 1 000 por clase— para que
la única diferencia entre ambos números sea la partición.

## Resultado

| | Partición aleatoria | Fuera de fuente |
|---|---:|---:|
| macro-F1 | 0,8411 | **0,5573** |
| Exactitud | 0,9115 | **0,6884** |
| Imágenes evaluadas | 5 015 | 33 268 |

**La partición honesta cuesta 0,284 de macro-F1.** Ese es el resultado de la fase, no una
regresión.

### Por clase

| Clase | Aleatoria | Fuera de fuente | Δ | Retiene | Banda |
|---|---:|---:|---:|---:|---|
| `healthy` | 0,9468 | 0,8092 | −0,138 | 85,5 % | Sostenida |
| `lethal_necrosis` | 0,9795 | 0,7912 | −0,188 | 80,8 % | Sostenida |
| `common_rust` | 0,9682 | 0,7804 | −0,188 | 80,6 % | Sostenida |
| `northern_corn_leaf_blight` | 0,8947 | 0,7171 | −0,178 | 80,2 % | Sostenida |
| `fall_armyworm` | 0,9054 | 0,6194 | −0,286 | 68,4 % | Frágil |
| `nitrogen_deficiency` | 0,7154 | 0,4190 | −0,296 | 58,6 % | Frágil |
| `gray_leaf_spot` | 0,8376 | 0,3972 | −0,441 | 47,4 % | No soportada |
| `phosphorus_deficiency` | 0,7390 | 0,2861 | −0,453 | 38,7 % | No soportada |
| `potassium_deficiency` | 0,5832 | 0,1958 | −0,387 | 33,6 % | No soportada |

La columna «Aleatoria» es la media de tres semillas del brazo `original` del test de fuga.
La columna «Fuera de fuente» es el F1 agrupado sobre las 33 268 predicciones, con una sola
semilla.

### Por pliegue

| Fuente retenida | Validación | n prueba | Clases | Exactitud |
|---|---|---:|---:|---:|
| `maize-diseases` | `maize-in-field-dataset` | 6 989 | 3 | 0,8891 |
| `maize-beans-tomatoes-africa` | `maize-deficiency-scanner-roboflow` | 12 140 | 4 | 0,7668 |
| `maize-deficiency-scanner-roboflow` | `maize-diseases` | 158 | 3 | 0,7405 |
| `multicrop-disease-maiz` | `corn-leaf-diseases-classification-roboflow` | 5 816 | 3 | 0,6816 |
| `corn-leaf-diseases-classification-roboflow` | `corn-leaf-roboflow` | 480 | 1 | 0,6744 |
| `maize-leaf-roboflow` | `maize-nutrient-deficiency` | 331 | 1 | 0,6707 |
| `cropdg-unified-multidomain` | `maize-2-roboflow` | 2 562 | 3 | 0,4328 |
| `corn-leaf-roboflow` | `cropdg-unified-multidomain` | 2 919 | 6 | 0,3875 |
| `maize-2-roboflow` | `maize-beans-tomatoes-africa` | 846 | 3 | 0,3527 |
| `maize-in-field-dataset` | `maize-leaf-roboflow` | 852 | 3 | 0,3032 |
| `maize-nutrient-deficiency` | `multicrop-disease-maiz` | 340 | 4 | 0,2136 |

El macro-F1 por pliegue **no es comparable entre pliegues** y por eso no se reporta aquí: los
pliegues cubren entre 1 y 6 clases, y el promedio macro se calcula sobre el conjunto de
etiquetas que aparecen en las predicciones, no sobre las presentes en la prueba. Un pliegue
de una sola clase, como `corn-leaf-diseases-classification-roboflow`, da macro-F1 0,0895 con
exactitud 0,6744. La cifra con sentido es la agrupada.

## Lectura

**Las cuatro clases que se sostienen son las que tienen volumen y varias fuentes.** Retienen
entre el 80 % y el 86 % de su F1 al cambiar de dominio, que es una caída esperable y
manejable.

**Las tres deficiencias nutricionales y `gray_leaf_spot` no sobreviven.** Retienen entre el
34 % y el 47 %. `potassium_deficiency` cae a 0,196: con 621 imágenes repartidas en cuatro
fuentes, cada pliegue le quita una fracción grande de un corpus ya pequeño.

**El caso de `gray_leaf_spot` es coherente con lo medido antes.** En el experimento del
segmentador esta clase se hundía de 0,945 a 0,204 cuando el recorte eliminaba el marco de la
imagen. Aquí cae de 0,838 a 0,397 cuando se le retira la fuente. Los dos experimentos
apuntan a lo mismo desde ángulos distintos: **el rendimiento de esta clase en la partición
aleatoria no proviene de la lesión**.

## El anillo de borde bajo el mismo protocolo

El criterio de la Compuerta 1 incluía una segunda condición sobre el anillo de borde, pero la
única medición disponible al redactar el plan se hizo sobre la partición **aleatoria**, donde
el atajo transfiere por construcción. Aplicarla a un F1 fuera de fuente mezcla dos protocolos
y no significa nada, así que **el brazo `border_ring` se volvió a ejecutar con los mismos once
pliegues**, cambiando únicamente lo que ve el modelo: un marco exterior del 10 %, sin hoja y
sin lesión.

```bash
modal run scripts/modal/leave_one_source_out.py --arm border_ring
```

| | Imagen completa | Solo el marco |
|---|---:|---:|
| macro-F1 agrupado | 0,5573 | **0,3870** |
| Exactitud agrupada | 0,6884 | **0,4894** |

**Aun con la partición honesta, el marco solo recupera el 69,5 % del macro-F1.** El atajo no
desaparece al separar las fuentes: se atenúa.

### Recuperación por clase

| Clase | Imagen completa | Solo el marco | El marco recupera |
|---|---:|---:|---:|
| `lethal_necrosis` | 0,7912 | 0,7889 | **99,7 %** |
| `common_rust` | 0,7804 | 0,7688 | **98,5 %** |
| `fall_armyworm` | 0,6194 | 0,4217 | 68,1 % |
| `nitrogen_deficiency` | 0,4190 | 0,2631 | 62,8 % |
| `northern_corn_leaf_blight` | 0,7171 | 0,4478 | 62,4 % |
| `healthy` | 0,8092 | 0,4534 | 56,0 % |
| `potassium_deficiency` | 0,1958 | 0,1062 | 54,2 % |
| `gray_leaf_spot` | 0,3972 | 0,1868 | 47,0 % |
| `phosphorus_deficiency` | 0,2861 | 0,0466 | 16,3 % |

**`lethal_necrosis` y `common_rust` siguen siendo identificables casi por completo desde el
marco aunque el modelo nunca haya visto su fuente.** Su F1 alto no mide capacidad
diagnóstica: mide que sus fuentes comparten una firma de captura entre ellas. Es coherente
con lo ya documentado: la mayor parte de `common_rust` son imágenes de laboratorio
pre-enmascaradas en negro, presentes en dos fuentes distintas derivadas de PlantVillage, y
`lethal_necrosis` procede de dos fuentes que son primeros planos a cuadro completo.

En el extremo opuesto, el F1 bajo de `phosphorus_deficiency` **no** proviene del marco: con
16,3 % de recuperación, esa clase simplemente es difícil y escasa.

## Compuerta 1

Con las dos condiciones medidas bajo el mismo protocolo:

| Clase | F1 fuera de fuente | El marco recupera | Banda |
|---|---:|---:|---|
| `healthy` | 0,8092 | 56,0 % | **Sostenida** |
| `lethal_necrosis` | 0,7912 | 99,7 % | Sostenida por procedencia |
| `common_rust` | 0,7804 | 98,5 % | Sostenida por procedencia |
| `northern_corn_leaf_blight` | 0,7171 | 62,4 % | Sostenida por procedencia |
| `fall_armyworm` | 0,6194 | 68,1 % | Frágil |
| `nitrogen_deficiency` | 0,4190 | 62,8 % | Frágil |
| `gray_leaf_spot` | 0,3972 | 47,0 % | No soportada |
| `phosphorus_deficiency` | 0,2861 | 16,3 % | No soportada |
| `potassium_deficiency` | 0,1958 | 54,2 % | No soportada |

### La compuerta produjo una celda que no había previsto

El plan definía tres bandas y exigía **ambas** condiciones para «sostenida», pero no decía
qué hacer con una clase que supera el umbral de F1 y **falla** el del marco. Tres clases caen
ahí. La resolución que se adopta, y que queda registrada como enmienda al criterio original:

> **Sostenida por procedencia.** F1 fuera de fuente ≥ 0,70 con recuperación del marco ≥ 60 %.
> La clase se predice bien, pero no se ha demostrado que sea por la lesión. Su número **no
> puede presentarse como capacidad diagnóstica** sin una de dos cosas: que una intervención de
> la Fase 2 baje la recuperación del marco por debajo del 60 % manteniendo el F1, o que se
> documente explícitamente como limitación.

Con esa enmienda, **una sola clase de nueve —`healthy`— supera la compuerta completa**, y con
poco margen: 56,0 % frente a un umbral de 60 %.

## Consolidación con tres semillas

La tabla anterior usa una sola semilla, y cuatro clases quedaban a menos de 2 σ de su umbral.
Se ejecutaron dos semillas más de **ambos** brazos —cuatro corridas de once pliegues, en
paralelo sobre cuatro A10— para obtener media y desviación **medidas bajo este protocolo**,
en lugar de importadas de la partición aleatoria.

```bash
modal run scripts/modal/leave_one_source_out.py --seed 1
modal run scripts/modal/leave_one_source_out.py --seed 2
modal run scripts/modal/leave_one_source_out.py --seed 1 --arm border_ring
modal run scripts/modal/leave_one_source_out.py --seed 2 --arm border_ring
```

| | Imagen completa | Solo el marco |
|---|---:|---:|
| macro-F1 agrupado | **0,5515 ± 0,0072** | **0,4001 ± 0,0137** |
| Exactitud agrupada | 0,6941 | — |
| El marco recupera | | **72,6 %** |

### Bandas con media ± σ

| Clase | F1 fuera de fuente | σ al umbral | El marco recupera | σ al umbral | Banda |
|---|---:|---:|---:|---:|---|
| `healthy` | 0,8192 ± 0,0099 | 12,0 | 62,4 % ± 5,7 | **0,4** | Sostenida por procedencia |
| `lethal_necrosis` | 0,8090 ± 0,0154 | 7,1 | 95,5 % ± 4,0 | 8,9 | Sostenida por procedencia |
| `common_rust` | 0,7847 ± 0,0407 | **2,1** | 104,1 % ± 8,6 | 5,1 | Sostenida por procedencia |
| `northern_corn_leaf_blight` | 0,7143 ± 0,0045 | 3,2 | 63,5 % ± 2,7 | **1,3** | Sostenida por procedencia |
| `fall_armyworm` | 0,6344 ± 0,0130 | 5,0 | 63,4 % ± 4,8 | — | Frágil |
| `nitrogen_deficiency` | 0,4387 ± 0,0171 | 2,3 | 61,6 % ± 8,3 | — | Frágil |
| `gray_leaf_spot` | 0,3709 ± 0,0263 | **1,1** | 55,0 % ± 7,0 | — | No soportada |
| `phosphorus_deficiency` | 0,2091 ± 0,0683 | 2,8 | 42,2 % ± 25,3 | — | No soportada |
| `potassium_deficiency` | 0,1831 ± 0,0110 | 19,7 | 51,1 % ± 5,6 | — | No soportada |

### Qué cambió respecto a una sola semilla

**Ninguna clase de las nueve supera la compuerta completa.** Con una semilla, `healthy` la
superaba con una recuperación del marco del 56,0 %. Con tres, su media sube a 62,4 % y cruza
el umbral: pasa a **sostenida por procedencia**. El resultado anterior era ruido, y queda
retirado.

**`common_rust` recupera el 104,1 % desde el marco**: el modelo que solo ve el borde predice
esa clase *mejor* que el que ve la imagen completa. Con 8,6 puntos de desviación, el exceso
sobre el 100 % no es significativo, pero la lectura sí lo es: para esta clase la hoja no
aporta nada por encima del contexto de captura.

**`gray_leaf_spot` bajó de 0,3972 a 0,3709 ± 0,0263.** Sigue a 1,1 σ del umbral, así que la
banda «no soportada» **no está firme**; simplemente es ahora la más probable de las dos.

**La desviación del cociente del marco, que antes no existía, resulta ser grande**: de 2,7
a 25,3 puntos porcentuales según la clase. En `phosphorus_deficiency` (± 25,3) el cociente es
directamente inutilizable, aunque su banda queda determinada por el F1, a 2,8 σ.

### Qué sigue sin estar firme

| Clase | Motivo | Consecuencia si cambia |
|---|---|---|
| `healthy` | 0,4 σ del umbral del marco | Es la única candidata a «sostenida» plena |
| `gray_leaf_spot` | 1,1 σ del umbral de F1 | Entre «no soportada» y «frágil» |
| `northern_corn_leaf_blight` | 1,3 σ del umbral del marco | Entre «sostenida» y «sostenida por procedencia» |
| `common_rust` | 2,1 σ del umbral de F1 | Entre «sostenida por procedencia» y «frágil» |

Tres semillas no bastan para estas cuatro. Se documentan como **indecidibles con el
protocolo actual** y, mientras no se resuelvan, se tratan con el criterio más conservador de
las dos bandas posibles.

## Qué entra en la Fase 2

Las dos clases frágiles (`fall_armyworm`, `nitrogen_deficiency`) y las tres sostenidas por
procedencia (`lethal_necrosis`, `common_rust`, `northern_corn_leaf_blight`), estas últimas
con un objetivo distinto: no subir el F1 sino **bajar la recuperación del marco** sin perderlo.

Las tres no soportadas pasan a limitación documentada, salvo que la Fase 2 las recupere por
encima de 0,40.

## Qué queda sin verificar

| Cuestión | Estado |
|---|---|
| `gray_leaf_spot` en 0,3972 | **En el límite** del umbral de 0,40 con una sola semilla. La banda asignada no es distinguible de «frágil»; requiere repetir con dos semillas más antes de darla por firme |
| Estabilidad del resto de bandas | Una sola semilla. Las cuatro sostenidas y las dos no soportadas restantes están lejos de sus umbrales, así que el riesgo de reclasificación es bajo, pero no está medido |
| Efecto del tope de entrenamiento | Se usó 1 000 por clase para igualar el protocolo del test de fuga. No se midió cuánto cambia el resultado sin tope |
| Casi-duplicados residuales | Se eliminaron los de hash idéntico (165). Los de distancia 1 a 4 siguen presentes, y su tasa de falsos positivos no está establecida (ver [Fase 0](/es/provenance/fase-0-auditoria)) |
| Independencia de las fuentes | 14 componentes de casi-duplicados cruzan grupos de procedencia con el criterio estricto, 9 de ellos entre `maize-diseases` y `multicrop-disease-maiz`. Los pliegues de esas dos fuentes están ligeramente contaminados entre sí |
