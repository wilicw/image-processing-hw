from PyQt5.QtWidgets import QApplication

import ui
from image_process import ImageProcess as processor
from cifar_inference import cifar_inference as vgg_processor

if __name__ == '__main__':
    app = QApplication([])
    window = ui.GraphicsInterface(processor(), vgg_processor())
    window.show()
    app.exec_()
