from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QVBoxLayout,
    QLabel,
    QFileDialog,
    QHBoxLayout,
    QGroupBox,
)
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import matplotlib.pyplot as plt


class GraphicsInterface(QWidget):
    def __init__(self, processor, mnist_processor, cat_dog_processor):
        super().__init__()
        self.processor = processor
        self.mnist_processor = mnist_processor
        self.cat_dog_processor = cat_dog_processor
        self.initUI()

    def mouseMoveEvent(self, event):
        painter = QPainter(self.prediction_board)
        painter.setPen(QPen(QColor(255, 255, 255), 10, Qt.SolidLine, Qt.RoundCap))
        painter.drawPoint(event.x(), event.y())
        self.prediction_image.setPixmap(self.prediction_board)

    def initUI(self):
        self.setWindowTitle("HW2")

        # Create a layout
        layout = QHBoxLayout()
        v1layout = QVBoxLayout()
        v2layout = QVBoxLayout()

        # Image Load
        load_layout = QVBoxLayout()
        load_button_1 = QPushButton("Load Image 1")
        load_button_1.clicked.connect(
            lambda: self.processor.add_image(self.load_image("image1"))
        )
        load_layout.addWidget(load_button_1)
        layout.addLayout(load_layout)

        # hw2-1
        hw2_1_layout = QVBoxLayout()
        hw2_1_gp = QGroupBox("1. Hough Circle Transform")
        draw_contour_button = QPushButton("1.1 Draw Contour")
        count_coin_button = QPushButton("1.2 Count Coins")
        coin_label = QLabel("There are _ conis in the image.")
        draw_contour_button.clicked.connect(self.processor.draw_conutor)
        count_coin_button.clicked.connect(
            lambda: coin_label.setText(
                "There are " + str(self.processor.count_coin()) + " conis in the image."
            )
        )
        hw2_1_layout.addWidget(draw_contour_button)
        hw2_1_layout.addWidget(count_coin_button)
        hw2_1_layout.addWidget(coin_label)
        hw2_1_gp.setLayout(hw2_1_layout)
        v1layout.addWidget(hw2_1_gp)

        # hw2-2
        hw2_2_layout = QVBoxLayout()
        hw2_2_gp = QGroupBox("2. Histogram Equalization")
        histogram_equalization_button = QPushButton("2. Histogram Equalization")
        histogram_equalization_button.clicked.connect(
            self.processor.histogram_equalization
        )
        hw2_2_layout.addWidget(histogram_equalization_button)
        hw2_2_gp.setLayout(hw2_2_layout)
        v1layout.addWidget(hw2_2_gp)

        # hw2-3
        hw2_3_layout = QVBoxLayout()
        hw2_3_gp = QGroupBox("3. Morphological Operations")
        closing_button = QPushButton("3.1 Closing")
        opening_button = QPushButton("3.2 Opening")
        closing_button.clicked.connect(self.processor.closing)
        opening_button.clicked.connect(self.processor.opening)
        hw2_3_layout.addWidget(closing_button)
        hw2_3_layout.addWidget(opening_button)
        hw2_3_gp.setLayout(hw2_3_layout)
        v1layout.addWidget(hw2_3_gp)

        # hw2-4
        hw2_4_layout = QVBoxLayout()
        hw2_4_gp = QGroupBox("4. MNIST Classification")
        model_structure_button = QPushButton("4.1 Show Model Structure")
        model_structure_button.clicked.connect(
            lambda: print(
                __import__("torchsummary").summary(
                    __import__("torchvision").models.vgg19_bn(num_classes=10),
                    input_size=(3, 32, 32),
                )
            )
        )
        acc_loss_button = QPushButton("4.2 Show Accuracy and Loss")
        acc_loss_button.clicked.connect(self.mnist_processor.accuracy_loss)
        inference_button = QPushButton("4.3 Predict")
        self.inference_result = QLabel()
        inference_button.clicked.connect(self.inference_callback)
        reset_button = QPushButton("4.4 Reset")
        reset_button.clicked.connect(
            lambda: self.prediction_board.fill(QColor(0, 0, 0))
            or self.prediction_image.setPixmap(self.prediction_board)
        )

        self.prediction_image = QLabel()
        self.prediction_image.setFixedSize(200, 200)

        self.prediction_board = QPixmap(200, 200)
        self.prediction_board.fill(QColor(0, 0, 0))
        self.prediction_image.setPixmap(self.prediction_board)

        self.prediction_image.mouseMoveEvent = self.mouseMoveEvent

        hw2_4_layout.addWidget(model_structure_button)
        hw2_4_layout.addWidget(acc_loss_button)
        hw2_4_layout.addWidget(inference_button)
        hw2_4_layout.addWidget(self.inference_result)
        hw2_4_layout.addWidget(reset_button)
        hw2_4_layout.addWidget(self.prediction_image)
        hw2_4_gp.setLayout(hw2_4_layout)
        v2layout.addWidget(hw2_4_gp)

        # hw2-5
        hw2_5_layout_main = QHBoxLayout()
        hw2_5_layout = QVBoxLayout()
        hw2_5_gp = QGroupBox("5. ResNet50")
        resnet_load_button = QPushButton("5.1 Load Image")
        resnet_show_button = QPushButton("5.2 Show Image")
        resnet_model_structure_button = QPushButton("5.3 Show Model Structure")
        resnet_show_comparison_button = QPushButton("5.4 Show Comparison")
        resnet_inference_button = QPushButton("5.5 Infernece")
        resnet_show_image = QLabel()
        self.resnet_predict_result = QLabel()
        resnet_show_image.setFixedSize(224, 224)
        resnet_load_button.clicked.connect(
            lambda: self.cat_dog_processor.add_image(self.load_image("image"))
            or resnet_show_image.setPixmap(
                QPixmap(
                    QImage(
                        self.cat_dog_processor.show_image,
                        self.cat_dog_processor.show_image.shape[1],
                        self.cat_dog_processor.show_image.shape[0],
                        3 * self.cat_dog_processor.show_image.shape[1],
                        QImage.Format_RGB888,
                    )
                )
            )
        )
        resnet_show_button.clicked.connect(self.cat_dog_processor.show_image)
        resnet_model_structure_button.clicked.connect(
            lambda: print(
                __import__("torchsummary").summary(
                    self.cat_dog_processor.model, input_size=(3, 224, 224)
                )
            )
        )
        resnet_show_comparison_button.clicked.connect(self.cat_dog_processor.compare)
        resnet_inference_button.clicked.connect(
            lambda: self.resnet_predict_result.setText(
                "Cat" if self.cat_dog_processor.inference() <= 0.5 else "Dog"
            )
        )

        hw2_5_layout.addWidget(resnet_load_button)
        hw2_5_layout.addWidget(resnet_show_button)
        hw2_5_layout.addWidget(resnet_model_structure_button)
        hw2_5_layout.addWidget(resnet_show_comparison_button)
        hw2_5_layout.addWidget(resnet_inference_button)

        hw2_5_layout_sub = QVBoxLayout()
        hw2_5_layout_sub.addWidget(resnet_show_image)
        hw2_5_layout_sub.addWidget(self.resnet_predict_result)

        hw2_5_layout_main.addLayout(hw2_5_layout)
        hw2_5_layout_main.addLayout(hw2_5_layout_sub)
        hw2_5_gp.setLayout(hw2_5_layout_main)
        v2layout.addWidget(hw2_5_gp)

        layout.addLayout(v1layout)
        layout.addLayout(v2layout)

        self.setLayout(layout)
        self.show()

    def load_image(self, image):
        return QFileDialog.getOpenFileName(
            self, "Open file", image, "Image files (*.jpg *.gif *.png)"
        )[0]

    def inference_callback(self):
        self.prediction_image.pixmap().toImage().save("tmp.png")
        self.mnist_processor.add_image("tmp.png")
        result = self.mnist_processor.inference()
        prediction = str(self.mnist_processor.classes[result[0].argmax()])
        self.inference_result.setText(prediction)
        plt.bar(self.mnist_processor.classes, result[0].detach().numpy())
        plt.title("Prediction Probability")
        plt.xlabel("Number")
        plt.ylabel("Probability")
        plt.show()
