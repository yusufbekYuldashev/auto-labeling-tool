import os
import sys
import glob
import random
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGridLayout, QDesktopWidget, QLineEdit
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSignal
from PIL import Image
from models.lang_sam.utils import draw_image
from ui.switch_dialog import SwitchDialog


class ImageDisplayApp(QWidget):
    proceed_signal = pyqtSignal(dict)

    def __init__(self, model, dataset_path, prompt):
        super().__init__()

        self.model = model
        self.dataset_path = dataset_path
        self.curr_prompt = prompt
        self.configs = {}

        self.dialog = None
        self.initUI()


    def initUI(self):
        # Get screen size
        screen = QDesktopWidget().screenGeometry()
        self.screen_width, self.screen_height = screen.width(), screen.height()

        # Define maximum image size
        self.image_width = self.screen_width // 2 - 20 # Two images horizontally with padding
        self.image_height = self.screen_height // 2 - 100  # Two images vertically with some space for buttons

        # Create layouts
        grid_layout = QGridLayout()
        button_layout = QHBoxLayout()        

        # Create buttons
        proceed_button = QPushButton('Proceed', self)
        proceed_button.setStyleSheet("background-color: green; color: white;")
        proceed_button.setFixedWidth(self.screen_width//3 - 10)
        proceed_button.clicked.connect(self.on_proceed)

        cancel_button = QPushButton('Cancel', self)
        cancel_button.setStyleSheet("background-color: red; color: white;")
        cancel_button.setFixedWidth(self.screen_width//3 - 10)
        cancel_button.clicked.connect(self.on_cancel)

        # Create a text input box
        self.text_input = QLineEdit(self)
        self.text_input.setFixedWidth(self.screen_width//3 - 10)
        self.text_input.returnPressed.connect(self.on_text_input)  # Connect Enter key event

        # Add buttons to the layout
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(self.text_input)
        button_layout.addWidget(proceed_button)        

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(grid_layout)
        main_layout.addLayout(button_layout)
  
        self.setLayout(main_layout)
        self.setWindowTitle('Image Display with Buttons')
        self.resize(self.screen_width, self.screen_height - 50)  # Adjust window height
        
        images = self.predict_for_display()
        self.display_images(images)
    
    
    def on_proceed(self):
        if self.dialog is None:
            self.dialog = SwitchDialog()
            self.dialog.generated.connect(self.on_generated)
            self.dialog.exec_()
    

    def on_generated(self, bbox_state, labelme_state):
        self.close()
        configs = {}
        configs['model'] = self.model
        configs['dataset_path'] = self.dataset_path
        configs['curr_prompt'] = self.curr_prompt
        configs['bbox'] = bbox_state
        configs['labelme'] = labelme_state
        self.proceed_signal.emit(configs)
        

    def on_cancel(self):
        sys.exit(0)


    def on_text_input(self):
        self.curr_prompt = self.text_input.text()  # Get the text from the input box
        self.display_images(self.predict_for_display())
        self.text_input.clear()  # Clear the input box after processing


    def predict_for_display(self):
        if not os.path.isdir(self.dataset_path):
            print('Dataset path is not a directory')
            return None
        
        all_img_paths = glob.glob(f'{self.dataset_path}/*.jpg')
        if len(all_img_paths) < 4:
            print('Not enough images are found in dataset')
            return None
        
        number_of_images = 4
        img_paths = random.choices(all_img_paths, k=number_of_images)
        images_to_process = [Image.open(img_path).convert('RGB') for img_path in img_paths]
        prompts_to_process = [self.curr_prompt for _ in range(number_of_images)]
        results = []
        for image, prompt in zip(images_to_process, prompts_to_process):
            result = self.model.predict([image], [prompt])
            results.append(result[0])

        result_images = []
        for result, image in zip(results, images_to_process):
            if result['masks'] is None or len(result['masks']) == 0:
                result_images.append(image)
                continue
            
            masks, xyxy, probs, labels = result['masks'], result['boxes'], result['scores'], result['labels']
            result_images.append(draw_image(Image.fromarray(np.array(image)), masks, xyxy, probs, labels))

        return [np.array(img) for img in result_images]


    def display_images(self, images):
        self.text_input.setPlaceholderText(f"Current prompt is {self.curr_prompt}")

        layout = self.layout().itemAt(0).layout()
        for i in reversed(range(layout.count())):  # Clear existing images
            layout.itemAt(i).widget().deleteLater()
        
        for i, image in enumerate(images):
            img = self.resize_and_convert(image)
            pixmap = QPixmap.fromImage(img)

            # Create a label to display the image
            label = QLabel(self)
            label.setPixmap(pixmap)
            label.setScaledContents(True)  # Allow the label to scale its contents
            layout.addWidget(label, i // 2, i % 2)  # 2x2 grid


    def resize_and_convert(self, image):
        """Open an image and adjust its orientation based on EXIF data."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        img = image.resize((self.image_width, self.image_height))  # Resize to new dimensions
        img = img.convert("RGBA")  # Convert to RGBA

        # Convert to QImage
        data = img.tobytes("raw", "RGBA")
        qimage = QImage(data, img.width, img.height, QImage.Format_RGBA8888)
        return qimage