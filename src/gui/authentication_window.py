# =====================================================================
#  IMPORTS
# =====================================================================

import os
import torch
import numpy as np

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QMessageBox, QFileDialog, QComboBox,
    QHBoxLayout, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEvent

from datasets.test import TripletSNN, CMUDatasetTriplet, embed_all
from sklearn.preprocessing import StandardScaler

from src.auth.security_controller import SecurityController
from src.snn.training_worker import TrainingWorker
from src.utils.data_utility import DataUtility
from src.auth.authenticator import Authenticator


# =====================================================================
#  CONSTANTS & GLOBAL CONFIG
# =====================================================================

MODEL_PATH = "models/snn_final.pt"


# =====================================================================
#  MAIN AUTHENTICATION WINDOW CLASS
# =====================================================================

class AuthenticationWindow(QWidget):

    # -----------------------------------------------------------------
    #  INITIALIZATION
    # -----------------------------------------------------------------
    def __init__(self):
        try:
            super().__init__()
            self.setWindowTitle("Authentication App")

            # State variables
            self.mode = "enrollment"
            self.enroll_target = 40
            self.enroll_count = 0
            self.enroll_filename = "enrollment_features.csv"
            self.enroll_append = True
            self.password_fixed = ".tie5Roanl"

            # Core helpers
            self.data_utility = DataUtility()
            self.security_controller = SecurityController(threshold=0.4)
            self.authenticator = Authenticator(threshold=0.4)

            # UI setup
            self.setup_layout()
            self.setup_enrollment_mode()

        except Exception as e:
            QMessageBox.critical(self, "Init Error", str(e))


    # =================================================================
    #  ENROLLMENT MODE
    # =================================================================
    def setup_enrollment_mode(self):
        try:
            if hasattr(self, "layout") and self.layout:
                self.clear_layout()

            self.resize(900, 650)
            self.center_on_screen()

            card = self.make_card()
            card_layout = self.setup_enrollment_card_layout(card)

            # Add form components
            self.add_card_layout_instructions(card_layout)
            self.add_card_username_input(card_layout)
            self.add_card_password_input(card_layout)
            self.add_card_enrollment_buttons(card_layout)
            self.add_card_enrollment_progress_label(card_layout)

            # Add card to main layout
            self.layout.addStretch()
            self.layout.addWidget(card, alignment=Qt.AlignCenter)
            self.layout.addStretch()

            self.data_utility.start()

        except Exception as e:
            QMessageBox.critical(self, "Enrollment Setup Error", str(e))


    def submit_enrollment_sample(self):
        try:
            username = self.username_entry.text()
            password = self.password_entry.text()

            # Validation
            if not username:
                QMessageBox.warning(self, "Enrollment", "Enter a username.")
                self.password_entry.clear()
                self.data_utility.reset()
                return

            if password != self.password_fixed:
                QMessageBox.warning(self, "Enrollment", "Password does not match.")
                self.password_entry.clear()
                self.data_utility.reset()
                return

            # Feature extraction
            self.data_utility.extract_features(username)
            filename = (
                self.enroll_filename
                if self.enroll_append else
                f"{self.enroll_count}_{self.enroll_filename}"
            )

            self.data_utility.save_features_csv(
                filename=filename,
                append=self.enroll_append
            )

            # Update progress
            self.enroll_count += 1
            self.progress_label.setText(
                f"Samples collected: {self.enroll_count} / {self.enroll_target}"
            )

            self.password_entry.clear()
            self.data_utility.reset()

            if self.enroll_count >= self.enroll_target:
                QMessageBox.information(
                    self,
                    "Enrollment Complete",
                    f"Collected {self.enroll_target} samples."
                )

        except Exception as e:
            QMessageBox.critical(self, "Enrollment Error", str(e))
    # =================================================================
    #  TRAINING MODE
    # =================================================================
    def setup_training_mode(self):
        self.clear_layout()
        self.resize(760, 480)
        self.center_on_screen()

        card = self.make_card()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(30, 30, 30, 30)
        card_layout.setSpacing(20)

        # Title row
        top_row = QHBoxLayout()
        top_row.addWidget(self.create_mode_selector(), alignment=Qt.AlignLeft)

        title = QLabel("Model Training")
        title.setObjectName("title")
        title.setFont(QFont("", 16, QFont.Bold))
        title.setStyleSheet("background: transparent;")
        title.setProperty("class", "title")

        top_row.addStretch()
        top_row.addWidget(title)
        top_row.addStretch()
        card_layout.addLayout(top_row)


        card_layout.addLayout(top_row)

        # Status label
        self.training_status = QLabel("Press the button below to start training.")
        self.training_status.setAlignment(Qt.AlignCenter)
        self.training_status.setProperty("class", "instr")
        card_layout.addWidget(self.training_status)

        # Button
        self.train_button = QPushButton("Start Training")
        self.train_button.setProperty("class", "primary")
        self.train_button.clicked.connect(self.start_model_training)
        card_layout.addWidget(self.train_button, alignment=Qt.AlignCenter)

        # Add card to UI
        self.layout.addStretch()
        self.layout.addWidget(card)
        self.layout.addStretch()



    def start_model_training(self):
        """Triggered when user presses 'Start Training'."""
        try:
            self.training_status.setText("Model is training... please wait.")
            self.train_button.setEnabled(False)

            from src.snn.model_trainer import ModelTrainer

            trainer = ModelTrainer(
                csv_path="datasets/ksenia_training_2.csv",
                out_dir="models",
                batch_size=64,
                lr=1e-3
            )

            self.worker = TrainingWorker(trainer)
            self.worker.finished.connect(self.on_training_finished)
            self.worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Training Error", str(e))

    def on_training_finished(self):
        self.training_status.setText("Training finished.")
        self.train_button.setEnabled(True)
        QMessageBox.information(self, "Training", "Model training finished successfully.")
    # =================================================================
    #  AUTHENTICATION MODE
    # =================================================================
    def setup_authentication_mode(self):
        self.clear_layout()
        self.resize(760, 480)
        self.center_on_screen()
        self.layout.setAlignment(Qt.AlignCenter)

        card = self.make_card()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 16, 18, 16)
        card_layout.setSpacing(12)

        # Title row
        top_row = QHBoxLayout()
        top_row.addWidget(self.create_mode_selector(), alignment=Qt.AlignLeft)

        title = QLabel("User Verification")
        title.setObjectName("title")
        title.setFont(QFont("", 12, QFont.Bold))
        title.setStyleSheet("background: transparent;")

        top_row.addStretch()
        top_row.addWidget(title)
        top_row.addStretch()
        card_layout.addLayout(top_row)

        # Username input
        urow = QHBoxLayout()
        self.username_label = QLabel("Username:")
        self.username_label.setProperty("class", "field-label")
        urow.addWidget(self.username_label, alignment=Qt.AlignVCenter)

        self.username_entry = QLineEdit()
        self.username_entry.setPlaceholderText("Enter your username")
        urow.addWidget(self.username_entry)
        card_layout.addLayout(urow)

        # Password input
        prow = QHBoxLayout()
        self.password_label = QLabel("Password:")
        self.password_label.setProperty("class", "field-label")
        prow.addWidget(self.password_label, alignment=Qt.AlignVCenter)

        self.password_entry = QLineEdit()
        self.password_entry.setEchoMode(QLineEdit.Password)
        self.password_entry.setPlaceholderText("Enter your password")
        self.password_entry.installEventFilter(self)
        prow.addWidget(self.password_entry)
        card_layout.addLayout(prow)

        # Authenticate button
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.authenticate_button = QPushButton("Authenticate")
        self.authenticate_button.setProperty("class", "primary")
        self.authenticate_button.clicked.connect(self.authenticate)
        btn_row.addWidget(self.authenticate_button)

        btn_row.addStretch()
        card_layout.addLayout(btn_row)

        # Result label
        self.result_label = QLabel("")
        self.result_label.setProperty("class", "status")
        self.result_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(self.result_label)

        # Add card to layout
        self.layout.addStretch()
        self.layout.addWidget(card, alignment=Qt.AlignCenter)
        self.layout.addStretch()
        self.setLayout(self.layout)

        self.data_utility.start()

    def authenticate(self):
        try:
            username = self.username_entry.text()
            password = self.password_entry.text()

            if not username or not password:
                QMessageBox.warning(self, "Authentication", "Fill all fields.")
                return

            try:
                self.data_utility.username = username
                self.data_utility.extract_features(username)
                features = self.data_utility.feature_extractor.key_features.data
                self.data_utility.save_features_csv(filename="temp_features.csv")
            except Exception as e:
                QMessageBox.critical(self, "Feature Error", str(e))
                self.password_entry.clear()
                return

            try:
                success, dist, message = self.authenticator.authenticate(username, password, features)
            except Exception as e:
                QMessageBox.critical(self, "Model Error", str(e))
                self.password_entry.clear()
                return

            if success:
                QMessageBox.information(self, "Authentication", f"{message}\nDistance = {dist:.4f}")
                try: self.switch_to_background_mode()
                except Exception as e:
                    QMessageBox.critical(self, "Background Mode Error", str(e))
                    self.password_entry.clear()
            else:
                QMessageBox.critical(self, "Authentication", f"{message}\nDistance = {dist:.4f}")
                self.password_entry.clear()

        except Exception as e:
            QMessageBox.critical(self, "Authentication Error", str(e))
            self.password_entry.clear()



    # =================================================================
    #  MODE SWITCHING
    # =================================================================
    def on_mode_changed(self, text):
        try:
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
                    self.authenticator.load_model(MODEL_PATH)
                except Exception as e:
                    QMessageBox.critical(self, "Model load failed", str(e))
                    return

                self.clear_layout()
                self.setup_authentication_mode()

        except Exception as e:
            QMessageBox.critical(self, "Mode Change Error", str(e))


    # =================================================================
    #  UI UTILITY FUNCTIONS
    # =================================================================
    def clear_layout(self):
        try:
            while self.layout.count():
                item = self.layout.takeAt(0)
                widget = item.widget()
                if widget:
                    if widget == getattr(self, "password_entry", None):
                        widget.removeEventFilter(self)
                    widget.deleteLater()
        except Exception as e:
            QMessageBox.critical(self, "Layout Clear Error", str(e))


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


    # =================================================================
    #  ENROLLMENT CARD SUBCOMPONENTS
    # =================================================================
    def setup_enrollment_card_layout(self, card):
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(30, 30, 30, 30)
        card_layout.setSpacing(20)

        # Title bar
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
            f"Please type your password exactly {self.enroll_target} times.\n"
            f"Password must be: {self.password_fixed}"
        )
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
        self.progress_label = QLabel(
            f"Samples collected: 0 / {self.enroll_target}"
        )
        self.progress_label.setProperty("class", "status")
        self.progress_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(self.progress_label)


    def add_card_enrollment_buttons(self, card_layout):
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        # Submit sample button
        self.enroll_button = QPushButton("Submit Sample")
        self.enroll_button.setProperty("class", "primary")
        self.enroll_button.clicked.connect(self.submit_enrollment_sample)
        btn_row.addWidget(self.enroll_button)

        # Load CSV button
        self.skip_enroll_button = QPushButton("Load CSV")
        self.skip_enroll_button.setProperty("class", "secondary")
        btn_row.addWidget(self.skip_enroll_button)

        btn_row.addStretch()
        card_layout.addLayout(btn_row)
