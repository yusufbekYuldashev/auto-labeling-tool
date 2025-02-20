from PyQt5.QtWidgets import (
    QVBoxLayout, QPushButton, QLabel, QDialog, QCheckBox, QHBoxLayout#, QSlider
)
from PyQt5.QtCore import Qt, pyqtSignal
from qtwidgets import Toggle


class SwitchDialog(QDialog):
    # Signal to send switch and tick states to the main window
    generated = pyqtSignal(bool, bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Label and shape configs")
        self.setGeometry(200, 200, 300, 200)

        # Initialize switch and tick states
        self.switch_state = True  # False means masks, True means bboxes
        self.tick_state = False  # Checkbox unchecked initially

        # Create a switch slider for toggle effect
        # self.switch_slider = QSlider(Qt.Horizontal, self)
        # self.switch_slider.setMinimum(0)
        # self.switch_slider.setMaximum(2)
        # self.switch_slider.setValue(1)  # Default: OFF
        # self.switch_slider.valueChanged.connect(self.on_switch_changed)
        self.switch_slider = Toggle()
        self.switch_slider.stateChanged.connect(self.on_switch_changed)

        # Create a tick checkbox
        self.tick_checkbox = QCheckBox("Labelme JSON", self)
        self.tick_checkbox.stateChanged.connect(self.on_tick_changed)

        # Generate button
        generate_button = QPushButton("Generate", self)
        generate_button.setStyleSheet("background-color: green; color: white;")
        generate_button.clicked.connect(self.on_generate)

        # Switch layout with "BBox" and "Masks" labels
        switch_layout = QHBoxLayout()
        
        bbox_label = QLabel("BBox", self)
        bbox_label.setAlignment(Qt.AlignCenter)
        
        masks_label = QLabel("Masks", self)
        bbox_label.setAlignment(Qt.AlignCenter)

        switch_layout.addWidget(bbox_label)
        switch_layout.addWidget(self.switch_slider)
        switch_layout.addWidget(masks_label)

        # Tick option layout
        tick_layout = QHBoxLayout()
        tick_layout.addWidget(self.tick_checkbox)

        # Main layout for the dialog
        dialog_layout = QVBoxLayout()
        dialog_layout.addLayout(switch_layout)
        dialog_layout.addLayout(tick_layout)
        dialog_layout.addWidget(generate_button)

        self.setLayout(dialog_layout)

    def on_switch_changed(self, value):
        """Update the switch state."""
        self.switch_state = value != Qt.Checked

    def on_tick_changed(self, state):
        """Update the tick state."""
        self.tick_state = state == Qt.Checked

    def on_generate(self):
        """Emit the states and close the dialog."""
        self.accept()  # Close the dialog
        self.generated.emit(self.switch_state, self.tick_state)
        self.close()

    def resizeEvent(self, event):
        """Adjust the switch slider width to 1/4 of the dialog width."""
        dialog_width = self.width()
        self.switch_slider.setFixedWidth(dialog_width // 3)  # Set slider width
        super().resizeEvent(event)  # Call the parent resize event