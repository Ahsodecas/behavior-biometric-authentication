import os

import torch
from PyQt5.QtWidgets import QApplication
from gui.authentication_window import AuthenticationWindow
import sys

def main():

    app = QApplication(sys.argv)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(base_dir, "gui", "css", "styles.css")

    # Load CSS
    with open(css_path, "r") as f:
        app.setStyleSheet(f.read())

    window = AuthenticationWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
