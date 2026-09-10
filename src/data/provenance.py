"""Identificación de la fuente de origen de cada imagen del corpus.

El corpus limpio sigue el convenio ``<clase>_<fuente>_<entorno>_<id>.<ext>``, así que la
procedencia es derivable del nombre sin trabajo de anotación. Varios tokens corresponden a
versiones del mismo dataset y se colapsan en un único grupo de procedencia: separarlos
permitiría que las mismas imágenes cayeran a ambos lados de una partición agrupada.
"""

from __future__ import annotations

import re

SOURCE_TO_DATASET = {
    "maize_field": "maize-in-field-dataset",
    "maize_desease": "maize-diseases",
    "maize_desease_v1.1": "maize-diseases",
    "cropdg": "cropdg-unified-multidomain",
    "maize_africa": "maize-beans-tomatoes-africa",
    "maize_africa_v1": "maize-beans-tomatoes-africa",
    "maize_africa_v1.2": "maize-beans-tomatoes-africa",
    "multi_desease": "multicrop-disease-maiz",
    "maize_nutrient": "maize-nutrient-deficiency",
    "corn_leaf_roboflow": "corn-leaf-roboflow",
    "maize_2_roboflow": "maize-2-roboflow",
    "maize_leaf_roboflow": "maize-leaf-roboflow",
    "maize_deficiency_scanner_roboflow": "maize-deficiency-scanner-roboflow",
    "corn_leaf_diseases_classification_roboflow": "corn-leaf-diseases-classification-roboflow",
}

PROVENANCE_GROUPS = sorted(set(SOURCE_TO_DATASET.values()))


def parse_source(class_name: str, environment: str, filename: str) -> str | None:
    """Extrae el token de fuente del nombre de archivo.

    Ancla el prefijo de clase y el sufijo de entorno porque los nombres de clase contienen
    guiones bajos y un patrón perezoso los partiría por el sitio equivocado.

    @param {str} class_name Clase a la que pertenece la imagen.
    @param {str} environment Entorno de captura, lab o real.
    @param {str} filename Nombre de archivo con extensión.
    @returns {str|None} Token de fuente, o None si el nombre no sigue el convenio.
    """
    stem = filename.rsplit(".", 1)[0]
    prefix = class_name + "_"
    if not stem.startswith(prefix):
        return None
    match = re.match(rf"^(?P<source>.+)_{environment}_[0-9a-fA-F]+$", stem[len(prefix):])
    return match.group("source") if match else None


def source_from_path(image_path: str) -> str | None:
    """Extrae el token de fuente desde una ruta ``clean/<clase>/<entorno>/<archivo>``.

    @param {str} image_path Ruta relativa a la raíz del dataset.
    @returns {str|None} Token de fuente, o None si la ruta no sigue la jerarquía esperada.
    """
    parts = image_path.replace("\\", "/").split("/")
    if len(parts) < 4:
        return None
    return parse_source(parts[-3], parts[-2], parts[-1])


def provenance_from_path(image_path: str) -> str | None:
    """Devuelve el grupo de procedencia (dataset de origen) de una ruta de imagen.

    @param {str} image_path Ruta relativa a la raíz del dataset.
    @returns {str|None} Nombre del dataset de origen, o None si no se pudo determinar.
    """
    source = source_from_path(image_path)
    return SOURCE_TO_DATASET.get(source) if source else None
