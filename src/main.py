import torch
from PyQt5.QtWidgets import QApplication
from gui.window import AuthWindow
import sys

def main():
    app = QApplication(sys.argv)
    window = AuthWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
