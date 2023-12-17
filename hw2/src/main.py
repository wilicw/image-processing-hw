from PyQt5.QtWidgets import QApplication

import ui
from image_process import ImageProcess as processor
from inference import mnist_inference as mnist_processor
from inference import cat_dog_inference as cat_dog_processor

if __name__ == '__main__':
    app = QApplication([])
    window = ui.GraphicsInterface(processor(), mnist_processor(), cat_dog_processor())
    window.show()
    app.exec_()
