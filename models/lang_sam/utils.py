import numpy as np
import supervision as sv
from PIL import Image


def load_image(image_path: str):
    return Image.open(image_path).convert("RGB")


def draw_image(image_rgb, masks, xyxy, probs, labels):
    box_annotator = sv.BoxCornerAnnotator()
    label_annotator = sv.LabelAnnotator()
    mask_annotator = sv.MaskAnnotator()
    # Create class_id for each unique label
    unique_labels = list(set(labels))
    class_id_map = {label: idx for idx, label in enumerate(unique_labels)}
    class_id = [class_id_map[label] for label in labels]

    # Add class_id to the Detections object
    detections = sv.Detections(
        xyxy=xyxy,
        mask=masks.astype(bool),
        confidence=probs,
        class_id=np.array(class_id),
    )
    annotated_image = box_annotator.annotate(scene=image_rgb, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    annotated_image = mask_annotator.annotate(scene=annotated_image, detections=detections)
    return annotated_image