from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QLabel,
    QFileDialog,
    QGridLayout,
    QLineEdit,
    QHBoxLayout,
    QGroupBox,
)


class GraphicsInterface(QWidget):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.initUI()

    def initUI(self):
        self.setWindowTitle("HW1")

        # Create a layout
        layout = QHBoxLayout()
        v1layout = QVBoxLayout()
        v2layout = QVBoxLayout()

        # Image Load
        load_layout = QVBoxLayout()
        load_button_1 = QPushButton("Load Image 1")
        load_button_2 = QPushButton("Load Image 2")
        load_button_1.clicked.connect(
            lambda: self.processor.add_image(self.load_image("image1"))
        )
        load_button_2.clicked.connect(
            lambda: self.processor.add_image(self.load_image("image2"))
        )
        load_layout.addWidget(load_button_1)
        load_layout.addWidget(load_button_2)
        layout.addLayout(load_layout)

        # hw1-1
        hw1_1_layout = QVBoxLayout()
        hw1_1_gp = QGroupBox("1. Image Processing")
        color_sep_button = QPushButton("1.1 Color Separation")
        color_tranformation_button = QPushButton("1.2 Color Transformation")
        color_extraction_button = QPushButton("1.3 Color Extraction")
        color_sep_button.clicked.connect(self.processor.color_separation)
        hw1_1_layout.addWidget(color_sep_button)
        color_tranformation_button.clicked.connect(self.processor.color_transformation)
        hw1_1_layout.addWidget(color_tranformation_button)
        hw1_1_layout.addWidget(color_extraction_button)
        hw1_1_gp.setLayout(hw1_1_layout)
        v1layout.addWidget(hw1_1_gp)

        # hw1-2
        hw1_2_layout = QVBoxLayout()
        hw1_2_gp = QGroupBox("2. Image Smoothing")
        guassian_button = QPushButton("2.1 Guassian Filter")
        bilateral_button = QPushButton("2.2 Bilateral Filter")
        median_button = QPushButton("2.3 Median Filter")
        hw1_2_layout.addWidget(guassian_button)
        hw1_2_layout.addWidget(bilateral_button)
        hw1_2_layout.addWidget(median_button)
        hw1_2_gp.setLayout(hw1_2_layout)
        v1layout.addWidget(hw1_2_gp)

        # hw1-3
        hw1_3_layout = QVBoxLayout()
        hw1_3_gp = QGroupBox("3. Edge Detection")
        sobelx_button = QPushButton("3.1 Sobel X")
        sobely_button = QPushButton("3.2 Sobel Y")
        combination_threshold_button = QPushButton("3.3 Combination and Threshold")
        gradient_angle_button = QPushButton("3.4 Gradient Angle")
        hw1_3_layout.addWidget(sobelx_button)
        hw1_3_layout.addWidget(sobely_button)
        hw1_3_layout.addWidget(combination_threshold_button)
        hw1_3_layout.addWidget(gradient_angle_button)
        hw1_3_gp.setLayout(hw1_3_layout)
        v1layout.addWidget(hw1_3_gp)

        # hw1-4
        hw1_4_layout = QVBoxLayout()
        hw1_4_gp = QGroupBox("4. Transforms")
        hw1_4_sub_layout = QHBoxLayout()
        hw1_4_label_layout = QVBoxLayout()
        hw1_4_label_layout.addWidget(QLabel("Rotation:"))
        hw1_4_label_layout.addWidget(QLabel("Scaling:"))
        hw1_4_label_layout.addWidget(QLabel("Tx:"))
        hw1_4_label_layout.addWidget(QLabel("Ty:"))
        hw1_4_sub_layout.addLayout(hw1_4_label_layout)

        hw1_4_input_layout = QVBoxLayout()
        rotation_input = QLineEdit()
        scaling_input = QLineEdit()
        tx_input = QLineEdit()
        ty_input = QLineEdit()
        hw1_4_input_layout.addWidget(rotation_input)
        hw1_4_input_layout.addWidget(scaling_input)
        hw1_4_input_layout.addWidget(tx_input)
        hw1_4_input_layout.addWidget(ty_input)
        hw1_4_sub_layout.addLayout(hw1_4_input_layout)
        hw1_4_layout.addLayout(hw1_4_sub_layout)
        apply_transform_button = QPushButton("4 Transforms")
        hw1_4_layout.addWidget(apply_transform_button)
        hw1_4_gp.setLayout(hw1_4_layout)
        v2layout.addWidget(hw1_4_gp)

        # hw1-5
        hw1_5_layout = QVBoxLayout()
        hw1_5_gp = QGroupBox("5. VGG19")
        vgg19_load_button = QPushButton("Load Image")
        arguementation_button = QPushButton("5.1 Show arguemented images")
        model_structure_button = QPushButton("5.2 Show Model Structure")
        acc_loss_button = QPushButton("5.3 Show Accuracy and Loss")
        inference_button = QPushButton("5.4 Inference")

        hw1_5_layout.addWidget(vgg19_load_button)
        hw1_5_layout.addWidget(arguementation_button)
        hw1_5_layout.addWidget(model_structure_button)
        hw1_5_layout.addWidget(acc_loss_button)
        hw1_5_layout.addWidget(inference_button)
        hw1_5_layout.addWidget(QLabel("Prediction:"))
        prediction_image = QLabel()
        prediction_image.setFixedSize(64, 64)
        hw1_5_layout.addWidget(prediction_image)

        hw1_5_gp.setLayout(hw1_5_layout)
        v2layout.addWidget(hw1_5_gp)

        layout.addLayout(v1layout)
        layout.addLayout(v2layout)

        self.setLayout(layout)
        self.show()

    def load_image(self, image):
        return QFileDialog.getOpenFileName(
            self, "Open file", image, "Image files (*.jpg *.gif *.png)"
        )[0]
