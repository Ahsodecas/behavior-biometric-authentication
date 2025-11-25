import os
import torch
import numpy as np
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QMessageBox, QFileDialog, QComboBox, QHBoxLayout, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEvent
from datasets.test import TripletSNN, CMUDatasetTriplet, embed_all
from sklearn.preprocessing import StandardScaler

from src.auth.security_controller import SecurityController
from src.utils.data_utility import DataUtility

model_path = "models/snn_final.pt"

class AuthenticationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Authentication App")

        # FIX change to enum
        self.mode = "enrollment"
        self.enroll_target = 40
        self.enroll_count = 0
        self.password_fixed = ".tie5Roanl"

        self.data_utility = DataUtility()
        self.security_controller = SecurityController(threshold=0.2)
        self.setup_layout()
        # FIX later should be changed to just login page and the security controller must initiate the right mode
        self.setup_enrollment_mode()

    def setup_enrollment_mode(self):
        if hasattr(self, "layout") and self.layout:
            self.clear_layout()

        self.resize(900, 650)
        self.center_on_screen()
        card = self.make_card()

        card_layout = self.setup_enrollment_card_layout(card)

        self.add_card_layout_instructions(card_layout)
        self.add_card_username_input(card_layout)
        self.add_card_password_input(card_layout)
        self.add_card_enrollment_buttons(card_layout)
        self.add_card_enrollment_progress_label(card_layout)

        self.layout.addStretch()
        self.layout.addWidget(card, alignment=Qt.AlignCenter)
        self.layout.addStretch()

        self.data_utility.start()

    def submit_enrollment_sample(self):
        username = self.username_entry.text()
        password = self.password_entry.text()

        if not username:
            QMessageBox.warning(self, "Enrollment", "Enter a username.")
            self.password_entry.clear()
            self.data_utility.reset()
            return

        if password != self.password_fixed:
            QMessageBox.warning(self, "Enrollment", "Password does not match the required one.")
            self.password_entry.clear()
            self.data_utility.reset()
            return

        self.data_utility.extract_features(username)
        self.data_utility.save_features_csv(filename=self.enroll_filename, append=False)
        self.enroll_count += 1
        self.progress_label.setText(f"Samples collected: {self.enroll_count} / {self.enroll_target}")

        self.password_entry.clear()
        self.data_utility.reset()

        if self.enroll_count >= self.enroll_target:
            QMessageBox.information(
                self, "Enrollment Complete",
                f"Collected {self.enroll_target} samples.\nSwitching to training phase."
            )
            # FIX implement, security module must switch to a different mode:
            # self.switch_to_training_mode()

    def setup_training_mode(self):
        # implement
        return None

    def setup_authentication_mode(self):
        # implement
        return None


    def on_mode_changed(self, text):
        if text == "Enrollment":
            self.mode = "enrollment"
            self.clear_layout()
            self.setup_enrollment_mode()

        elif text == "Training":
            self.mode = "training"
            self.clear_layout()
            self.setup_training_mode()

        elif text == "Authentication":
            self.mode = "authentication"
            try:
                self.load_snn_model(model_path)
            except Exception as e:
                QMessageBox.critical(self, "Model load failed", f"Failed to load model:\n{e}")
                return
            self.clear_layout()
            self.setup_authentication_mode()


    def create_mode_selector(self):
        box = QComboBox()
        box.addItems(["Enrollment", "Training", "Authentication"])
        box.setCurrentText({
                               "enrollment": "Enrollment",
                               "training": "Training",
                               "authentication": "Authentication"
                           }.get(self.mode, "Enrollment"))
        box.currentTextChanged.connect(self.on_mode_changed)
        return box

    def eventFilter(self, obj, event):
        if obj == self.password_entry:
            if event.type() == QEvent.KeyPress:
                self.data_utility.feed_key_event(event, "press")
            elif event.type() == QEvent.KeyRelease:
                self.data_utility.feed_key_event(event, "release")

        return super().eventFilter(obj, event)

    def setup_layout(self):
        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignCenter)
        self.layout.setSpacing(20)
        self.setLayout(self.layout)

    def center_on_screen(self):
        frame = self.frameGeometry()
        screen = QApplication.primaryScreen().availableGeometry().center()
        frame.moveCenter(screen)
        self.move(frame.topLeft())

    def make_card(self):
        card = QFrame()
        card.setObjectName("card")
        card.setMinimumWidth(600)
        card.setMinimumHeight(400)
        return card

    def setup_enrollment_card_layout(self, card):
        card_layout = QVBoxLayout(card)

        card_layout.setContentsMargins(30, 30, 30, 30)
        card_layout.setSpacing(20)

        # Title and mode selector
        top_row = QHBoxLayout()
        mode_box = self.create_mode_selector()
        mode_box.setFixedWidth(180)
        top_row.addWidget(mode_box, alignment=Qt.AlignLeft)

        title = QLabel("User Enrollment")
        title.setObjectName("title")
        title.setProperty("class", "title")
        title.setAlignment(Qt.AlignCenter)
        top_row.addStretch()
        top_row.addWidget(title)
        top_row.addStretch()
        card_layout.addLayout(top_row)

        return card_layout

    def add_card_layout_instructions(self, card_layout):
        instr = QLabel(
            f"Please type your password exactly {self.enroll_target} times.\nPassword must be: {self.password_fixed}")
        instr.setProperty("class", "instr")
        instr.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(instr)

    def add_card_username_input(self, card_layout):
        username_row = QHBoxLayout()
        username_label = QLabel("Username:")
        username_label.setProperty("class", "field-label")
        username_row.addWidget(username_label, alignment=Qt.AlignVCenter)
        self.username_entry = QLineEdit()
        self.username_entry.setPlaceholderText("Choose a username")
        username_row.addWidget(self.username_entry)
        card_layout.addLayout(username_row)

    def add_card_password_input(self, card_layout):
        password_row = QHBoxLayout()
        password_label = QLabel("Password:")
        password_label.setProperty("class", "field-label")
        password_row.addWidget(password_label, alignment=Qt.AlignVCenter)
        self.password_entry = QLineEdit()
        self.password_entry.setEchoMode(QLineEdit.Password)
        self.password_entry.setPlaceholderText("Enter the password")
        self.password_entry.installEventFilter(self)
        password_row.addWidget(self.password_entry)
        card_layout.addLayout(password_row)

    def add_card_enrollment_progress_label(self, card_layout):
        self.progress_label = QLabel(f"Samples collected: 0 / {self.enroll_target}")
        self.progress_label.setProperty("class", "status")
        self.progress_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(self.progress_label)

    def add_card_enrollment_buttons(self, card_layout):
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.enroll_button = QPushButton("Submit Sample")
        self.enroll_button.setProperty("class", "primary")
        self.enroll_button.clicked.connect(self.submit_enrollment_sample)  # <--- ADD THIS
        btn_row.addWidget(self.enroll_button)

        self.skip_enroll_button = QPushButton("Load CSV")
        self.skip_enroll_button.setProperty("class", "secondary")
        btn_row.addWidget(self.skip_enroll_button)
        btn_row.addStretch()

        card_layout.addLayout(btn_row)

    def clear_layout(self):
        try:
            while self.layout.count():
                item = self.layout.takeAt(0)
                widget = item.widget()
                if widget:
                    if widget == getattr(self, "password_entry", None):
                        widget.removeEventFilter(self)
                    widget.deleteLater()
        except Exception:
            pass
