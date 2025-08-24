from typing import Any, Dict, List

from yolo_detection.yolo_detector import YOLODetector


class YOLOWrapper:
    def __init__(self, model_size: str = "s", base_confidence: float = 0.4) -> None:
        self.detector = YOLODetector(model_size=model_size, base_confidence=base_confidence)

    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        result = self.detector.analyze_photo(image_path)
        if result is None:
            return {"object_counts": {}, "detections": [], "top_objects": [], "person_boxes": []}

        # Collect top objects by confidence
        detections = result.get("detections", [])
        # Add center coordinates to bbox for convenience
        for d in detections:
            bbox = d.get("bbox", {})
            if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
                x1 = float(bbox.get("x1", 0.0))
                y1 = float(bbox.get("y1", 0.0))
                x2 = float(bbox.get("x2", 0.0))
                y2 = float(bbox.get("y2", 0.0))
                bbox["center_x"] = (x1 + x2) / 2.0
                bbox["center_y"] = (y1 + y2) / 2.0
        top_sorted = sorted(detections, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
        top_objects = [
            {"class_name": d.get("class_name"), "confidence": float(d.get("confidence", 0.0))}
            for d in top_sorted[:10]
        ]

        # Extract person boxes for CLIP gender classification
        person_boxes: List[List[float]] = []
        for d in detections:
            if d.get("class_name") == "person":
                bbox = d.get("bbox", {})
                person_boxes.append([
                    float(bbox.get("x1", 0.0)),
                    float(bbox.get("y1", 0.0)),
                    float(bbox.get("x2", 0.0)),
                    float(bbox.get("y2", 0.0)),
                ])

        return {
            "object_counts": result.get("object_counts", {}),
            "detections": detections,
            "top_objects": top_objects,
            "person_boxes": person_boxes,
        }


