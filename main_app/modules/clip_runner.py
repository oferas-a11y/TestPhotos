from typing import Any, Dict, List
from pathlib import Path

import torch
import clip  # type: ignore
from PIL import Image


class CLIPManager:
    def __init__(self, model_name: str = "ViT-B/32") -> None:
        # Device selection
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model_name = model_name
        # Load once per run
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

        # Background category prompts (focus on environment, not activities)
        self.background_categories: List[str] = [
            "building interior",
            "church interior",
            "house interior",
            "room interior",
            "street",
            "town square",
            "bridge",
            "train station",
            "field",
            "forest",
            "park",
            "river",
            "lake",
            "mountain",
            "beach"
        ]
        self.background_templates: List[str] = [
            "the background shows {}",
            "an old photograph background of {}",
            "{} in the background"
        ]
        self.background_text_features = self._encode_prompt_set(self.background_categories, self.background_templates)

        # Indoor/Outdoor binary classifier prompts
        self.io_labels = ["indoor background", "outdoor background"]
        self.io_text_features = self._encode_prompt_set(self.io_labels, ["{}"])

    # (Scene analysis removed; using background_analysis instead)

    # ---------- Gender on person crops ----------
    def gender_on_person_crops(self, image_path: str, person_boxes: List[List[float]]) -> List[Dict[str, Any]]:
        if not person_boxes:
            return []
        img = Image.open(image_path).convert('RGB')
        w, h = img.size

        labels = ["man", "woman"]
        text = clip.tokenize(labels).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        results: List[Dict[str, Any]] = []
        for i, (x1, y1, x2, y2) in enumerate(person_boxes, 1):
            # Clamp and crop
            x1i, y1i = max(0, int(x1)), max(0, int(y1))
            x2i, y2i = min(w, int(x2)), min(h, int(y2))
            if x2i <= x1i or y2i <= y1i:
                continue
            crop = img.crop((x1i, y1i, x2i, y2i))
            image_input = self.preprocess(crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = (image_features @ text_features.T).squeeze(0)
                # Convert cosine to [0,1]
                probs = ((logits.clamp(-1, 1) + 1.0) / 2.0).tolist()
            results.append({
                "person_index": i,
                "bbox": [x1, y1, x2, y2],
                "man": float(probs[0]),
                "woman": float(probs[1])
            })
        return results

    # (Child vs Adult omitted per request)

    # ---------- Background analysis (replaces generic scene) ----------
    def _region_crops(self, image: Image.Image) -> List[Image.Image]:
        w, h = image.size
        crops: List[Image.Image] = []
        s = int(min(w, h) * 0.7)
        cx, cy = w // 2, h // 2
        crops.append(image.crop((cx - s//2, cy - s//2, cx + s//2, cy + s//2)))
        crops.append(image.crop((0, 0, w//2, h//2)))
        crops.append(image.crop((w//2, 0, w, h//2)))
        crops.append(image.crop((0, h//2, w//2, h)))
        crops.append(image.crop((w//2, h//2, w, h)))
        return crops

    def _encode_prompt_set(self, labels: List[str], templates: List[str]) -> torch.Tensor:
        texts: List[str] = []
        for lab in labels:
            for t in templates:
                texts.append(t.format(lab))
        with torch.no_grad():
            tok = clip.tokenize(texts).to(self.device)
            feats = self.model.encode_text(tok)
            feats /= feats.norm(dim=-1, keepdim=True)
        # Average per label block
        per_label: List[torch.Tensor] = []
        idx = 0
        for _ in labels:
            block = feats[idx: idx + len(templates)]
            avg = block.mean(dim=0)
            avg /= avg.norm()
            per_label.append(avg)
            idx += len(templates)
        return torch.stack(per_label, dim=0)

    def background_analysis(self, image_path: str, top_k: int = 8) -> Dict[str, Any]:
        try:
            img = Image.open(image_path).convert('RGB')
            # Region pooling to capture environment
            crops = self._region_crops(img)
            inputs = torch.stack([self.preprocess(c) for c in [img] + crops], dim=0).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(inputs)
                feats /= feats.norm(dim=-1, keepdim=True)
                # Max-pool over regions to emphasize background cues
                img_feat = feats.max(dim=0, keepdim=True)[0]

                # Indoor/Outdoor
                io_sims = (img_feat @ self.io_text_features.T).squeeze(0)
                io_sims = io_sims.clamp(-1, 1)
                io_scores = ((io_sims + 1.0) / 2.0).tolist()
                indoor_score, outdoor_score = float(io_scores[0]), float(io_scores[1])
                indoor_outdoor = 'indoor' if indoor_score >= outdoor_score else 'outdoor'

                # Background categories
                bg_sims = (img_feat @ self.background_text_features.T).squeeze(0)
                bg_sims = bg_sims.clamp(-1, 1)
                bg_scores = ((bg_sims + 1.0) / 2.0)
                scores_np = bg_scores.detach().cpu().numpy()
                order = scores_np.argsort()[::-1]
                detections: List[Dict[str, Any]] = []
                for idx in order[:top_k]:
                    detections.append({
                        'category': self.background_categories[int(idx)],
                        'confidence': float(scores_np[int(idx)])
                    })

                return {
                    'indoor_outdoor': indoor_outdoor,
                    'indoor_score': indoor_score,
                    'outdoor_score': outdoor_score,
                    'detections': detections,
                    'top_categories': [d['category'] for d in detections[:5]]
                }
        except Exception:
            pass
        return {
            'indoor_outdoor': 'unknown',
            'detections': [],
            'top_categories': []
        }

    # ---------- Army presence (separate activation) ----------
    def assess_army_presence(self, image_path: str) -> Dict[str, Any]:
        """Binary zero-shot: soldiers (army) vs civilian group."""
        try:
            img = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(img).unsqueeze(0).to(self.device)
            labels = [
                "soldiers in military uniforms",
                "civilian group without uniforms"
            ]
            with torch.no_grad():
                text = clip.tokenize(labels).to(self.device)
                txt = self.model.encode_text(text)
                txt /= txt.norm(dim=-1, keepdim=True)
                imgf = self.model.encode_image(image_input)
                imgf /= imgf.norm(dim=-1, keepdim=True)
                logits = (imgf @ txt.T).squeeze(0)
                # Convert to [0,1]
                probs = ((logits.clamp(-1, 1) + 1.0) / 2.0).tolist()
                soldier_score = float(probs[0])
                civilian_score = float(probs[1])
                is_army = soldier_score >= civilian_score
                return {
                    'is_army': bool(is_army),
                    'soldier_score': soldier_score,
                    'civilian_score': civilian_score
                }
        except Exception:
            pass
        return {'is_army': False}


