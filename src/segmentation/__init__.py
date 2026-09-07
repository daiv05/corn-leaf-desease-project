"""Módulo de segmentación de hojas de maíz."""

from src.segmentation.detector import MaizeLeafSegmenter
from src.segmentation.geometry import (
    apply_leaf_mask,
    binary_mask_image,
    crop_leaf_region,
    letterbox_image,
)
from src.segmentation.leaf_processor import (
    CROP_MASK_BLACK,
    CROP_MASK_LETTERBOX,
    MASK_BLACK,
    LeafInstance,
    LeafMaskProcessorConfig,
    SegmentedLeafProcessingResult,
    SegmentedLeafProcessor,
    build_comparison_panel,
)

__all__ = [
    "MaizeLeafSegmenter",
    "apply_leaf_mask",
    "binary_mask_image",
    "crop_leaf_region",
    "letterbox_image",
    "LeafInstance",
    "LeafMaskProcessorConfig",
    "SegmentedLeafProcessingResult",
    "SegmentedLeafProcessor",
    "build_comparison_panel",
    "MASK_BLACK",
    "CROP_MASK_BLACK",
    "CROP_MASK_LETTERBOX",
]

