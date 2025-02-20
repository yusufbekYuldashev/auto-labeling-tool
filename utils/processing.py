import os
import glob
import json
import tqdm
import shutil
from utils.helpers import generate_labelme_json, predict_single_image, create_yolo_label_file, create_yolo_data_yaml


def process_all_images(configs):
    model = configs['model']
    dataset_path = configs['dataset_path']
    curr_prompt = configs['curr_prompt']
    bbox = configs['bbox']
    labelme = configs['labelme']

    yolo_dataset_name = 'YOLO_BBOX_DATASET/' if bbox else 'YOLO_MASK_DATASET/'
    yolo_dataset_path = os.path.join(dataset_path, yolo_dataset_name)
    yolo_imgs_folder = os.path.join(yolo_dataset_path, 'train/images')
    yolo_labels_folder = os.path.join(yolo_dataset_path, 'train/labels')
    os.makedirs(yolo_dataset_path, exist_ok=True)
    os.makedirs(yolo_imgs_folder, exist_ok=True)
    os.makedirs(yolo_labels_folder, exist_ok=True)
    
    if labelme:
        labelme_labels_folder = dataset_path
    
    jpg_img_paths = glob.glob(f"{dataset_path}/*.jpg")
    png_img_paths = glob.glob(f"{dataset_path}/*.png")
    all_img_paths = jpg_img_paths + png_img_paths
    print(f"\nThere are {len(all_img_paths)} images in the given directory. Creating yolo {',labelme' if labelme else ''} {'bbox' if bbox else 'mask'} label files using prompt '{curr_prompt}'.........")
    
    label_to_class = {}
    class_counter = 0
    for img_path in tqdm.tqdm(all_img_paths):
        # Predicting masks, bboxes and labels for the img_path
        masks, bboxes, labels, img_size = predict_single_image(img_path, curr_prompt, model)
        
        # Generating class_id -> label dictionary
        class_ids = []
        if labels is not None:
            for label in labels:
                if not label in label_to_class.keys():
                    class_ids.append(class_counter)
                    label_to_class[label] = class_counter
                    class_counter += 1
                else:
                    class_ids.append(label_to_class[label])
        
        # Copying image to yolo image folder and creating .txt file
        shutil.copy(img_path, yolo_imgs_folder)
        yolo_label_path = os.path.join(yolo_labels_folder, os.path.basename(img_path)[:-4] + '.txt')  
        create_yolo_label_file(yolo_label_path, class_ids, img_size, masks, bboxes, is_bbox=bbox)

        # If labelme flag was set, generating json file with annotation in labelme format
        if labelme:
            labelme_label_path = os.path.join(labelme_labels_folder, os.path.basename(img_path)[:-4] + '.json')
            labelme_json = generate_labelme_json(masks, bboxes, labels, img_size, os.path.basename(img_path), is_bbox=bbox)
            with open(labelme_label_path, 'w') as f:
                json.dump(labelme_json, f, indent=4)

    # Creating data.yaml file for yolo dataset
    create_yolo_data_yaml(yolo_imgs_folder, label_to_class, yolo_dataset_path)