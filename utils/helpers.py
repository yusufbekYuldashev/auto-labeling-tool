import os
import yaml
import cv2
import numpy as np
from PIL import Image


MIN_AREA = 100


def get_contours(mask):
    if len(mask.shape) > 2:
        mask = np.squeeze(mask, 0)
    mask = mask.astype(np.uint8)
    mask *= 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    effContours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_AREA:
            effContours.append(c)
    return effContours


def contour_to_points(contour):
    pointsNum = len(contour)
    contour = contour.reshape(pointsNum, -1).astype(np.float32)
    points = [point.tolist() for point in contour]
    return points


def generate_labelme_json(binary_masks, bboxes, labels, image_size, image_path=None, is_bbox=False):
    """Generate a LabelMe format JSON file from binary mask tensor.

    Args:
        binary_masks: Binary mask tensor of shape [N, H, W].
        labels: List of labels for each mask.
        image_size: Tuple of (height, width) for the image size.
        image_path: Path to the image file (optional).

    Returns:
        A dictionary representing the LabelMe JSON file.
    """
    json_dict = {
        "version": "4.5.6",
        "imageHeight": image_size[1],
        "imageWidth": image_size[0],
        "imagePath": image_path,
        "flags": {},
        "shapes": [],
        "imageData": None,
    }
    if bboxes is None:
        return json_dict
    
    if is_bbox:
        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = [float(coord) for coord in bbox]
            shape_dict = {
                'label': str(label),
                'points': [[x1, y1], [x2, y2]],
                'group_id': None,
                'shape_type': 'rectangle',
                'flags': {}
            }
            json_dict['shapes'].append(shape_dict)

    elif not is_bbox:
        # Loop through the masks and add them to the JSON dictionary
        num_masks = binary_masks.shape[0]
        # binary_masks = binary_masks.numpy()
        for i in range(num_masks):
            mask = binary_masks[i]
            label = labels[i]
            effContours = get_contours(mask)

            for effContour in effContours:
                points = contour_to_points(effContour)
                shape_dict = {
                    "label": label,
                    "line_color": None,
                    "fill_color": None,
                    "points": points,
                    "shape_type": "polygon",
                }

                json_dict["shapes"].append(shape_dict)

    return json_dict


def create_yolo_data_yaml(yolo_imgs_target, label_to_class, yolo_dataset_path):
    yaml_data = {
        'train': f'{os.path.abspath(yolo_imgs_target)}',
        'nc': len(label_to_class.keys()),
        'names': {l2c[1]: l2c[0] for l2c in label_to_class.items()}
    }
    with open(os.path.join(yolo_dataset_path, 'data.yaml'), 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)


def bbox_to_yolo(size, bbox):
    width, height = size
    x1, y1, x2, y2 = bbox
    
    x_center = (x1 + x2) / 2.0 / width
    y_center = (y1 + y2) / 2.0 / height
    bbox_width = (x2 - x1) / width
    bbox_height = (y2 - y1) / height

    return [x_center, y_center, bbox_width, bbox_height]


def mask_to_yolo(size, mask):
    width, height = size
    
    mask = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    points = largest_contour.squeeze(axis=1)
    normalized = [(x/width, y/height) for (x, y) in points]
    return [coord for point in normalized for coord in point]


def create_yolo_label_file(label_path, class_ids, img_size, masks=None, boxes=None, is_bbox=False):
    with open(label_path, 'w') as f:
        if len(class_ids) > 0:
            if is_bbox:
                for class_id, bbox in zip(class_ids, boxes):
                    yolo_bbox = bbox_to_yolo(img_size, bbox)
                    f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")
            else:
                for class_id, mask in zip(class_ids, masks):
                    rle = mask_to_yolo(img_size, mask)
                    rle_str = ' '.join(map(str, rle))
                    f.write(f"{class_id} {rle_str}\n")
        f.close()


def predict_single_image(img_path, prompt, model):
    img = Image.open(img_path).convert('RGB')
    result = model.predict([img], [prompt])[0]
    if result['masks'] is None or len(result['masks']) == 0:
        return None, None, None, img.size
    return result['masks'], result['boxes'], result['labels'], img.size