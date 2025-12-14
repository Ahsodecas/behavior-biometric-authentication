# =====================================================================
#  IMPORTS
# =====================================================================

import os
import csv
from datetime import time

import torch
import numpy as np

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QMessageBox, QFileDialog, QComboBox,
    QHBoxLayout, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEvent


from src.ml.data_preprocessor import DataPreprocessor
from src.ml.training_worker import TrainingWorker
from src.utils.data_utility import DataUtility
from src.auth.authentication_decision_maker import AuthenticationDecisionMaker
from src.auth.background_auth_manager import BackgroundAuthManager
from src.ml.model_trainer import ModelTrainer


# =====================================================================
#  CONSTANTS & GLOBAL CONFIG
# =====================================================================


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(ROOT_DIR))
PATH_MODELS = os.path.join(PROJECT_ROOT, "models")
PATH_DATASETS = os.path.join(PROJECT_ROOT, "datasets")
PATH_EXTRACTED = os.path.join(PROJECT_ROOT, "extracted_features")


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

            # ---------------- State ----------------
            self.mode = "landing"  # landing | enrollment | training | authentication
            self.username = None

            self.enroll_target = 40
            self.enroll_count = 0
            self.enroll_filename = "enrollment_features.csv"
            self.enroll_append = True
            self.password_fixed = ".tie5Roanl"

            # ---------------- Core helpers ----------------
            self.data_utility = DataUtility()
            self.authenticator = AuthenticationDecisionMaker(threshold=0.3)

            # UI setup
            self.setup_layout()
            self.setup_landing_page()

        except Exception as e:
            QMessageBox.critical(self, "Init Error", str(e))

    # =====================================================
    # LANDING PAGE (USERNAME + LOGIN / REGISTER)
    # =====================================================
    def setup_landing_page(self):
        self.clear_layout()
        self.resize(700, 450)
        self.center_on_screen()

        card = self.make_card()
        card.setMinimumWidth(520)
        card.setMinimumHeight(320)

        layout = QVBoxLayout(card)
        layout.setSpacing(24)
        layout.setContentsMargins(40, 40, 40, 40)

        # Title
        title = QLabel("Keystroke Authentication")
        title.setProperty("class", "title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel(
            "Secure user verification using behavioral biometrics"
        )
        subtitle.setProperty("class", "instr")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        # Username field
        self.username_entry = QLineEdit()
        self.username_entry.setPlaceholderText("Enter your username")
        self.username_entry.setObjectName("landing-username")
        layout.addWidget(self.username_entry)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(16)

        login_btn = QPushButton("Login")
        login_btn.setProperty("class", "primary")
        login_btn.clicked.connect(self.handle_login)

        register_btn = QPushButton("Register")
        register_btn.setProperty("class", "secondary")
        register_btn.clicked.connect(self.handle_register)

        btn_row.addStretch()
        btn_row.addWidget(login_btn)
        btn_row.addWidget(register_btn)
        btn_row.addStretch()

        layout.addLayout(btn_row)

        self.layout.addStretch()
        self.layout.addWidget(card, alignment=Qt.AlignCenter)
        self.layout.addStretch()

    def handle_register(self):
        username = self.username_entry.text().strip()
        if not username:
            QMessageBox.warning(self, "Register", "Please enter a username")
            return

        self.username = username
        self.authenticator.username = self.username
        self.enroll_count = 0
        self.mode = "enrollment"
        self.setup_enrollment_mode()

    def handle_login(self):
        username = self.username_entry.text().strip()
        if not username:
            QMessageBox.warning(self, "Login", "Please enter a username")
            return

        model_path = os.path.join(PATH_MODELS, f"{username}_snn.pt")
        if not os.path.exists(model_path):
            QMessageBox.information(
                self,
                "No Model",
                "No trained model found. Please register first."
            )
            return

        self.username = username
        self.authenticator.username = username
        self.mode = "authentication"
        self.authenticator.load_model(model_path)
        self.setup_authentication_mode()


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
        password = self.password_entry.text()

        if password != self.password_fixed:
            QMessageBox.warning(self, "Enrollment", "Password does not match")
            self.password_entry.clear()
            self.data_utility.reset(failed=True)
            return

        self.data_utility.extract_features(self.username)
        self.data_utility.save_features_csv(self.enroll_filename, append=True)

        self.enroll_count += 1
        self.progress_label.setText(
            f"Samples collected: {self.enroll_count} / {self.enroll_target}"
        )

        self.password_entry.clear()
        self.data_utility.reset()

        if self.enroll_count >= self.enroll_target:
            QMessageBox.information(self, "Enrollment", "Enrollment complete")
            self.mode = "training"
            self.setup_training_mode()

    def load_csv_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Features CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return

        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Load CSV", "Selected file does not exist.")
            return

        metadata = {}
        features = {}
        username = ""

        try:
            username = self.data_utility.feature_extractor.key_features.load_csv_features_all_rows(file_path)
            QMessageBox.information(self, "Load CSV", f"Features successfully loaded from {file_path}.")

            #print("Features read: ")
            #print(self.data_utility.feature_extractor.key_features.all_features)
            filename = (
                self.enroll_filename
                if self.enroll_append else
                f"{self.enroll_count}_{self.enroll_filename}"
            )
            self.data_utility.generate_synthetic_features(username, filename, repetitions=10)

        except Exception as e:
            QMessageBox.critical(self, "Load CSV", f"Failed to load features:\n{str(e)}")


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

            self.training_status.setText("Data is processing.... Please wait.")
            self.train_button.setEnabled(False)

            username = self.username

            preprocessor = DataPreprocessor(
                enrollment_csv=os.path.join(PATH_EXTRACTED, username, f"enrollment_features.csv"),
                dsl_dataset_csv=os.path.join(PATH_DATASETS, "DSL-StrongPasswordData.csv"),
                username=username,
                output_csv=os.path.join(PATH_DATASETS, f"{username}_training.csv"),
                synth_reps=0
            )

            trainer = ModelTrainer(
                csv_path=os.path.join(PATH_DATASETS, f"{username}_training.csv"),
                out_dir=PATH_MODELS,
                username=username,
                batch_size=64,
                lr=1e-3
            )

            print(f"1")
            self.worker = TrainingWorker(trainer, preprocessor, username)
            self.worker.dataProcFinished.connect(self.on_data_processing_finished)
            self.worker.trainFinished.connect(self.on_training_finished)
            self.worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Training Error", str(e))

    def on_training_finished(self):
        self.training_status.setText("Training finished.")
        self.train_button.setEnabled(False)
        QMessageBox.information(self, "Training", "Model training finished successfully.")
        self.on_mode_changed("Authentication")

    def on_data_processing_finished(self):
        self.training_status.setText("Data Processing finished. Training... Please wait.")
        self.train_button.setEnabled(False)
    # =================================================================
    #  AUTHENTICATION MODE
    # =================================================================
    def setup_authentication_mode(self):
        print("Setting up authentication mode...")
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
                self.username = username
                self.data_utility.username = username
                self.data_utility.mouse_data_collector.username = username
                self.data_utility.extract_features(username)
                features = self.data_utility.feature_extractor.key_features.data
                self.data_utility.save_raw_csv(filename="raw.csv")
                self.data_utility.save_features_csv(filename="temp_features.csv")
            except Exception as e:
                QMessageBox.critical(self, "Feature Error", str(e))
                self.data_utility.reset(failed=True)
                self.password_entry.clear()
                return

            try:
                success, dist, message = self.authenticator.authenticate(username, password, features)
            except Exception as e:
                QMessageBox.critical(self, "Model Error", str(e))
                self.data_utility.reset(failed=True)
                self.password_entry.clear()
                return

            if success:
                QMessageBox.information(self, "Authentication", f"{message}\nDistance = {dist:.4f}")
                try: self.switch_to_background_mode()
                except Exception as e:
                    QMessageBox.critical(self, "Background Mode Error", str(e))

                self.data_utility.reset()
                self.password_entry.clear()
            else:
                QMessageBox.critical(self, "Authentication", f"{message}\nDistance = {dist:.4f}")

                self.data_utility.reset(failed=True)
                self.password_entry.clear()

        except Exception as e:
            QMessageBox.critical(self, "Authentication Error", str(e))

            self.data_utility.reset(failed=True)
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
                    self.authenticator.load_model(os.path.join(PATH_MODELS, f"{self.username}_snn.pt"))
                except Exception as e:
                    QMessageBox.critical(self, "Model load failed", str(e))
                    return

                self.clear_layout()
                self.setup_authentication_mode()

        except Exception as e:
            QMessageBox.critical(self, "Mode Change Error", str(e))


    def switch_to_background_mode(self):
        # ---- UI setup errors ----
        try:
            if hasattr(self, "bg_window") and self.bg_window:
                self.close_background_mode()

            self.bg_window = QWidget()
            self.bg_window.setObjectName("bg-card")
            self.bg_window.setFixedSize(280, 180)
            self.bg_window.setWindowFlags(
                Qt.WindowStaysOnTopHint |
                Qt.FramelessWindowHint |
                Qt.Tool
            )

            layout = QVBoxLayout(self.bg_window)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(10)

            title = QLabel("Background Service")
            title.setProperty("class", "bg-title")
            layout.addWidget(title)

            row = QHBoxLayout()
            status = QLabel("Status")
            status.setProperty("class", "status")
            row.addWidget(status)
            row.addStretch()

            self.bg_status_label = QLabel("● Idle")
            self.bg_status_label.setProperty("class", "status-pill idle")
            row.addWidget(self.bg_status_label)
            layout.addLayout(row)

            result_row = QHBoxLayout()

            result_label = QLabel("Last auth")
            result_label.setProperty("class", "status")
            result_row.addWidget(result_label)
            result_row.addStretch()

            self.bg_result_label = QLabel("— Waiting")
            self.bg_result_label.setProperty("class", "result-pill neutral")
            result_row.addWidget(self.bg_result_label)

            layout.addLayout(result_row)

            logout_btn = QPushButton("Logout")
            logout_btn.setProperty("class", "danger")
            logout_btn.clicked.connect(self.close_background_mode)
            layout.addWidget(logout_btn)

            screen = QApplication.primaryScreen().availableGeometry()
            self.bg_window.move(
                screen.right() - self.bg_window.width() - 20,
                screen.bottom() - self.bg_window.height() - 20
            )

            self.bg_window.show()
            self.hide()

        except Exception as e:
            QMessageBox.critical(self, "UI Error", f"Failed to start background UI:\n{e}")
            return

        try:
            if hasattr(self, "bg_manager") and self.bg_manager:
                self.bg_manager.stop()

            self.bg_manager = BackgroundAuthManager(
                username=self.username,
                data_utility=self.data_utility,
                authenticator_model_path=os.path.join(
                    PATH_MODELS, f"{self.username}_cnn_mouse.keras"
                ),
            )

            self.bg_manager.status_update.connect(self.bg_status_label.setText)
            self.bg_manager.auth_result.connect(self.on_auth_result)
            self.bg_manager.start()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Background Auth Error",
                f"Failed to start background authentication:\n{e}"
            )
            self.close_background_mode()

    def close_background_mode(self):
        try:
            if hasattr(self, "bg_manager") and self.bg_manager:
                try:
                    self.bg_manager.stop()
                except Exception:
                    pass

                self.bg_manager.deleteLater()
                self.bg_manager = None

            if hasattr(self, "bg_result_label"):
                self.bg_result_label.setText("— Waiting")

            if hasattr(self, "bg_window") and self.bg_window:
                self.bg_window.close()
                self.bg_window = None



            self.setup_authentication_mode()
            self.show()

        except Exception as e:
            QMessageBox.critical(self, "Logout Error", str(e))

    def on_auth_result(self, accepted: bool, mean_score: float):
        try:
            if not self.bg_manager:
                return

            if accepted:
                self.bg_status_label.setText("● Authenticated")

                self.bg_result_label.setText(f"✓ Accepted ({mean_score:.2f})")
                self.bg_result_label.setProperty("class", "result-pill success")
                self.bg_result_label.style().unpolish(self.bg_result_label)
                self.bg_result_label.style().polish(self.bg_result_label)

                self.data_utility.clear_mouse_data()
                print(f"Authentication Successful. Score: {mean_score}")

                QTimer.singleShot(1000, self.bg_manager.start)

            else:
                self.bg_result_label.setText(f"✗ Rejected ({mean_score:.2f})")
                self.bg_result_label.setProperty("class", "result-pill danger")
                self.bg_result_label.style().unpolish(self.bg_result_label)
                self.bg_result_label.style().polish(self.bg_result_label)

                self.bg_manager.stop()
                QMessageBox.warning(
                    self,
                    "Authentication Failed",
                    "Behavioral authentication failed. Please authenticate again."
                )
                self.close_background_mode()

        except Exception as e:
            QMessageBox.critical(self, "Background Auth Error", str(e))
            self.close_background_mode()

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
        self.skip_enroll_button.clicked.connect(self.load_csv_data)
        self.skip_enroll_button.setProperty("class", "primary")
        btn_row.addWidget(self.skip_enroll_button)

        btn_row.addStretch()
        card_layout.addLayout(btn_row)
