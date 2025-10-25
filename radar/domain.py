from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np


@dataclass
class RadarParameters:
    fc: float = 300e6
    cc: float = 3e8
    bw: float = 7e6
    fs: Optional[float] = None
    tp: Optional[float] = None

    def __post_init__(self):
        if self.fs is None:
            self.fs = self.bw * 1 / 10000
        if self.tp is None:
            self.tp = 50 / self.fs


@dataclass
class ImageData:
    base_image: Optional[np.ndarray] = None
    processed_image: Optional[np.ndarray] = None
    overlay_image: Optional[np.ndarray] = None
    polar_image: Optional[np.ndarray] = None
    clean_generated: Optional[np.ndarray] = None
    is_generated: bool = False
    has_noise: bool = False

    def update_processed(self, image: np.ndarray):
        self.processed_image = image
        if self.is_generated:
            self.has_noise = True


@dataclass
class DetectionResult:
    bbox: Tuple[int, int, int, int]
    class_name: str
    confidence: float