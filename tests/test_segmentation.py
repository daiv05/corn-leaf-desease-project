"""Pruebas unitarias para el módulo de segmentación y enmascaramiento."""

import numpy as np
from PIL import Image

from src.segmentation.geometry import (
    apply_leaf_mask,
    binary_mask_image,
    clip_bbox,
    crop_leaf_region,
    letterbox_image,
    mask_area_ratio,
    mask_bbox,
)
from src.segmentation.leaf_processor import (
    CROP_MASK_LETTERBOX,
    LeafInstance,
    LeafMaskProcessorConfig,
    SegmentedLeafProcessor,
    build_comparison_panel,
)


def test_geometry_and_mask_operations() -> None:
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    mask = Image.new("L", (100, 100), 0)
    # Dibujar un cuadrado blanco de 40x40 en el centro
    mask_arr = np.array(mask)
    mask_arr[30:70, 30:70] = 255
    mask = Image.fromarray(mask_arr, mode="L")

    # Ratio de área esperado: 1600 / 10000 = 0.16
    ratio = mask_area_ratio(mask)
    assert abs(ratio - 0.16) < 1e-4

    bbox = mask_bbox(mask)
    assert bbox == (30, 30, 70, 70)

    clipped = clip_bbox((25.5, 25.5, 75.2, 75.8), 100, 100)
    assert clipped == (25, 25, 76, 76)

    cropped = crop_leaf_region(img, (30, 30, 70, 70))
    assert cropped.size == (40, 40)

    # Letterbox a 224x224
    lbox = letterbox_image(cropped, (224, 224))
    assert lbox.image.size == (224, 224)
    assert lbox.target_size == (224, 224)

    # Enmascarado
    masked = apply_leaf_mask(img, mask, background_value=(0, 0, 0))
    masked_np = np.array(masked)
    assert masked_np[0, 0].tolist() == [0, 0, 0]
    assert masked_np[50, 50].tolist() == [255, 255, 255]


def test_segmented_leaf_processor() -> None:
    img = Image.new("RGB", (100, 100), (120, 200, 100))
    mask = Image.new("L", (100, 100), 0)
    mask_arr = np.array(mask)
    mask_arr[20:80, 20:80] = 255
    mask = Image.fromarray(mask_arr, mode="L")

    instance = LeafInstance(
        mask=mask,
        confidence=0.92,
        bbox=(20, 20, 80, 80),
        source_index=0,
        class_id=0,
    )

    config = LeafMaskProcessorConfig(
        processing_profile=CROP_MASK_LETTERBOX,
        target_size=(224, 224),
    )
    processor = SegmentedLeafProcessor(config=config)
    result = processor.process(img, [instance])

    assert not result.fallback_used
    assert result.processed_image is not None
    assert result.processed_image.size == (224, 224)

    # Panel de preview
    panel = build_comparison_panel(result)
    assert panel.size[1] == 240

