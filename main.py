import sys
import argparse
from PyQt5.QtWidgets import QApplication
from models.lang_sam import LangSAM
from utils.processing import process_all_images
from ui.image_display import ImageDisplayApp


def parse_args():
    parser = argparse.ArgumentParser(description='Predict masks and boxes for dataset and save in yolo format')
    parser.add_argument('--dataset_path', type=str, help='Path to the images', required=True)
    parser.add_argument('--prompt', type=str, help='Prompt for segmentation', required=True)
    return parser.parse_args()


def main():
    app = QApplication(sys.argv)
    
    args = parse_args()
    model = LangSAM()

    display_app = ImageDisplayApp(model, args.dataset_path, args.prompt)
    display_app.proceed_signal.connect(process_all_images)
    display_app.show()


    sys.exit(app.exec_())


if __name__ == '__main__':
    main()