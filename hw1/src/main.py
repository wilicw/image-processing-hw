# pyqt5 simple ui
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QGridLayout, QComboBox

import ui
from image_process import ImageProcess as processor
from cifar_vgg19 import cifar_vgg as vgg_processor

if __name__ == '__main__':
    app = QApplication([])
    window = ui.GraphicsInterface(processor(), vgg_processor())
    window.show()
    app.exec_()
