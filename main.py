import sys
from my_gui import MyGui
from PyQt5.QtWidgets import QApplication


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = MyGui()
    gui.show()
    sys.exit(app.exec_())
