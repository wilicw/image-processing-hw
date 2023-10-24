# pyqt5 simple ui
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QGridLayout, QComboBox

import ui

if __name__ == '__main__':
    app = QApplication([])
    window = ui.GraphicsInterface()
    window.show()
    app.exec_()
