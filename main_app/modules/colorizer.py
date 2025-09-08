import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np  # type: ignore
import cv2  # type: ignore


class Colorizer:
    def __init__(self, models_dir: str) -> None:
        self.models_dir = models_dir
        self._net: Optional[Any] = None
        self._pts_path: Optional[str] = None

    def _ensure_model_files(self) -> None:
        prototxt = os.path.join(self.models_dir, "colorization_deploy_v2.prototxt")
        caffemodel = os.path.join(self.models_dir, "colorization_release_v2.caffemodel")
        pts = os.path.join(self.models_dir, "pts_in_hull.npy")
        missing = [p for p in [prototxt, caffemodel, pts] if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                "Missing model files. Expected: \n"
                f"- {prototxt}\n"
                f"- {caffemodel}\n"
                f"- {pts}\n"
                "Please download them (see README) and try again."
            )
        size_bytes = os.path.getsize(caffemodel)
        if size_bytes < 100_000_000:
            raise RuntimeError(
                f"caffemodel seems too small ({size_bytes} bytes). Re-download from an official mirror."
            )

    def _ensure_loaded(self) -> None:
        if self._net is not None:
            return
        self._ensure_model_files()
        prototxt = os.path.join(self.models_dir, "colorization_deploy_v2.prototxt")
        caffemodel = os.path.join(self.models_dir, "colorization_release_v2.caffemodel")
        pts_path = os.path.join(self.models_dir, "pts_in_hull.npy")
        net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        pts_in_hull = np.load(pts_path)
        pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
        net.getLayer(net.getLayerId("class8_ab")).blobs = [pts_in_hull.astype(np.float32)]
        net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]
        self._net = net
        self._pts_path = pts_path

    def _colorize_path(self, input_path: str, output_path: str, ab_boost: float) -> None:
        self._ensure_loaded()
        assert self._net is not None
        image_bgr = cv2.imread(input_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {input_path}")
        image_bgr_float = image_bgr.astype(np.float32) / 255.0
        image_lab = cv2.cvtColor(image_bgr_float, cv2.COLOR_BGR2LAB)
        image_l = image_lab[:, :, 0]
        image_l_rs = cv2.resize(image_l, (224, 224))
        image_l_rs = image_l_rs - 50
        blob = cv2.dnn.blobFromImage(image_l_rs)
        self._net.setInput(blob)
        ab_dec = self._net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab_dec_us = cv2.resize(ab_dec, (image_bgr.shape[1], image_bgr.shape[0]))
        if ab_boost != 1.0:
            ab_dec_us = ab_dec_us * float(ab_boost)
        ab_dec_us = np.clip(ab_dec_us, -128.0, 127.0)
        lab_out = np.concatenate((image_l[:, :, np.newaxis], ab_dec_us), axis=2).astype(np.float32)
        image_bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
        image_bgr_out = np.clip(image_bgr_out, 0.0, 1.0)
        image_bgr_out = (image_bgr_out * 255.0).astype(np.uint8)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image_bgr_out)

    def colorize_directory(self, input_directory: str, output_directory: str, ab_boost: float = 1.0) -> Dict[str, str]:
        """Colorize all supported images in a directory. Returns map: original_filename -> colorized_path."""
        in_dir = Path(input_directory)
        out_dir = Path(output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".JPG", ".JPEG", ".PNG"]
        mapping: Dict[str, str] = {}

        for p in sorted(in_dir.iterdir()):
            if p.suffix in extensions and p.is_file():
                out_path = out_dir / f"colorized_{p.stem}{p.suffix}"
                self._colorize_path(str(p), str(out_path), ab_boost=ab_boost)
                mapping[p.name] = str(out_path)

        return mapping

    def colorize_files(self, files: List[str], output_directory: str, ab_boost: float = 1.0) -> Dict[str, str]:
        """Colorize a specific list of files. Returns map: original_full_path -> colorized_path."""
        out_dir = Path(output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        mapping: Dict[str, str] = {}
        for fp in files:
            p = Path(fp)
            if not p.exists() or not p.is_file():
                continue
            out_path = out_dir / f"colorized_{p.stem}{p.suffix}"
            self._colorize_path(str(p), str(out_path), ab_boost=ab_boost)
            mapping[str(p)] = str(out_path)
        return mapping


