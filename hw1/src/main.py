# pyqt5 simple ui
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QGridLayout, QComboBox

import ui
from image_process import ImageProcess as processor

if __name__ == '__main__':
    app = QApplication([])
    window = ui.GraphicsInterface(processor())
    window.show()
    app.exec_()
