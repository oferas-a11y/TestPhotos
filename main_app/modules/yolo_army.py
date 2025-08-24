from typing import Any, Dict, List, Optional


class YOLOArmyDetector:
    """Conditional YOLO model for army-related objects (uniform, rifle, cannon, camp).

    This expects a custom YOLO weights file trained to predict these classes.
    If weights are missing or the model cannot be loaded, `available` is False and
    `analyze_image` returns empty counts.
    """

    def __init__(self, weights_path: str = "yolo_army.pt", confidence: float = 0.35, imgsz: int = 1280) -> None:
        self.available = False
        self.model = None  # type: ignore[assignment]
        self.confidence = confidence
        self.imgsz = imgsz
        self.target_classes = ["uniform", "rifle", "cannon", "camp"]
        try:
            from ultralytics import YOLO  # type: ignore
            import os
            if os.path.isfile(weights_path):
                self.model = YOLO(weights_path)
                # Device auto
                try:
                    import torch  # type: ignore
                    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        self.device = 'mps'
                    elif torch.cuda.is_available():
                        self.device = 'cuda'
                    else:
                        self.device = 'cpu'
                    try:
                        self.model.to(self.device)
                    except Exception:
                        pass
                except Exception:
                    self.device = 'cpu'  # type: ignore[assignment]
                self.available = True
        except Exception:
            self.available = False

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        if not self.available or self.model is None:
            return {"object_counts": {c: 0 for c in self.target_classes}, "detections": []}
        try:
            results = self.model(
                image_path,
                conf=self.confidence,
                imgsz=self.imgsz,
                max_det=200
            )
            counts: Dict[str, int] = {c: 0 for c in self.target_classes}
            detections: List[Dict[str, Any]] = []
            if results and len(results) > 0:
                r = results[0]
                if getattr(r, 'boxes', None) is not None and len(r.boxes) > 0:  # type: ignore[attr-defined]
                    boxes = r.boxes.xyxy.cpu().numpy()  # type: ignore[attr-defined]
                    confs = r.boxes.conf.cpu().numpy()  # type: ignore[attr-defined]
                    clsi = r.boxes.cls.cpu().numpy().astype(int)  # type: ignore[attr-defined]
                    names = r.names if hasattr(r, 'names') else {}
                    for (x1, y1, x2, y2), cf, ci in zip(boxes, confs, clsi):
                        cname = str(names.get(int(ci), str(ci)))
                        if cname in counts:
                            counts[cname] += 1
                            detections.append({
                                'class_name': cname,
                                'confidence': float(cf),
                                'bbox': {
                                    'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)
                                }
                            })
            return {"object_counts": counts, "detections": detections}
        except Exception:
            return {"object_counts": {c: 0 for c in self.target_classes}, "detections": []}


