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
    QHBoxLayout, QFrame, QSizePolicy, QDialog, QMainWindow, QTextEdit
)

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QEvent, QTimer
from numpy.f2py.crackfortran import usermodules

from sklearn.preprocessing import StandardScaler

from src.ml.data_preprocessor import DataPreprocessor
from src.ml.mouse_model_trainer import MouseModelTrainer
from src.ml.training_worker import TrainingWorker
from src.utils.data_utility import DataUtility
from src.auth.authentication_decision_maker import AuthenticationDecisionMaker
from src.auth.background_auth_manager import BackgroundAuthManager
from src.ml.model_trainer import ModelTrainer
from src.utils.user_management_utility import UserManagementUtility
from src.utils.logger import Logger


# =====================================================================
#  CONSTANTS & GLOBAL CONFIG
# =====================================================================

import src.constants as constants

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
            self.username = ""

            # ---------------- Core helpers ----------------
            self.data_utility = DataUtility()
            self.authenticator = AuthenticationDecisionMaker(threshold=0.1)
            self.user_management_utility = UserManagementUtility()
            self.logger = Logger()

            if not self.superuser_exists():
                dialog = self.SuperuserDialog()
                if dialog.exec_() == QDialog.Accepted:
                    username, password = dialog.get_credentials()
                    self.user_management_utility.create_superuser(username, password)

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

        superuser_btn = QPushButton("Login Superuser")
        superuser_btn.setProperty("class", "secondary")
        superuser_btn.clicked.connect(self.handle_login_superuser)

        btn_row.addStretch()
        btn_row.addWidget(login_btn)
        btn_row.addWidget(register_btn)
        btn_row.addWidget(superuser_btn)
        btn_row.addStretch()

        layout.addLayout(btn_row)

        self.layout.addStretch()
        self.layout.addWidget(card, alignment=Qt.AlignCenter)
        self.layout.addStretch()

    def handle_register(self):
        try:
            username = self.username_entry.text().strip()
            if not username:
                QMessageBox.warning(self, "Register", "Please enter a username")
                return

            self.username = username
            self.data_utility.set_username(username)
            self.authenticator.username = self.username

            self.mode = "set_password"
            self.setup_set_password_page()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during registration:\n{str(e)}")

    def handle_login(self):
        try:
            username = self.username_entry.text().strip()
            if not username:
                QMessageBox.warning(self, "Login", "Please enter a username")
                return

            model_path = os.path.join(constants.PATH_MODELS, f"{username}_snn.pt")
            if not os.path.exists(model_path):
                QMessageBox.information(
                    self,
                    "No Model",
                    "No trained model found. Please register first."
                )
                return

            self.username = username
            self.authenticator.username = username
            self.data_utility.set_username(username)
            self.mode = "authentication"
            self.authenticator.load_model(model_path, username=username, training_csv=f"{self.username}_training.csv")
            self.setup_authentication_mode()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during login:\n{str(e)}")
            print(f"handle_login error: {e}")

    def setup_set_password_page(self):
        try:
            self.clear_layout()
            self.resize(700, 450)
            self.center_on_screen()

            card = self.make_card()
            card.setMinimumWidth(520)

            layout = QVBoxLayout(card)
            layout.setContentsMargins(30, 30, 30, 30)
            layout.setSpacing(16)

            title = QLabel("Set Your Password")
            title.setProperty("class", "title")
            title.setAlignment(Qt.AlignCenter)
            layout.addWidget(title)

            subtitle = QLabel("Create a password for your new account")
            subtitle.setProperty("class", "instr")
            subtitle.setAlignment(Qt.AlignCenter)
            layout.addWidget(subtitle)

            self.new_password_entry = QLineEdit()
            self.new_password_entry.setEchoMode(QLineEdit.Password)
            self.new_password_entry.setPlaceholderText("Enter new password")
            layout.addWidget(self.new_password_entry)

            self.confirm_password_entry = QLineEdit()
            self.confirm_password_entry.setEchoMode(QLineEdit.Password)
            self.confirm_password_entry.setPlaceholderText("Confirm password")
            layout.addWidget(self.confirm_password_entry)

            btn_row = QHBoxLayout()
            btn_row.setSpacing(16)

            save_btn = QPushButton("Save Password")
            save_btn.setProperty("class", "primary")
            save_btn.clicked.connect(self.save_password_and_continue)

            cancel_btn = QPushButton("Cancel")
            cancel_btn.setProperty("class", "secondary")
            cancel_btn.clicked.connect(self.setup_landing_page)

            btn_row.addStretch()
            btn_row.addWidget(save_btn)
            btn_row.addWidget(cancel_btn)
            btn_row.addStretch()

            layout.addLayout(btn_row)

            self.layout.addStretch()
            self.layout.addWidget(card, alignment=Qt.AlignCenter)
            self.layout.addStretch()

        except Exception as e:
            QMessageBox.critical(self, "Password Setup Error", str(e))

    def save_password_and_continue(self):
        pwd1 = self.new_password_entry.text()
        pwd2 = self.confirm_password_entry.text()

        if not pwd1 or not pwd2:
            QMessageBox.warning(self, "Password", "Please fill both fields.")
            return

        if pwd1 != pwd2:
            QMessageBox.warning(self, "Password", "Passwords do not match.")
            return

        constants.PASSWORD = pwd1
        self.user_management_utility.create_local_user(self.username, pwd1)

        QMessageBox.information(self, "Password Set", "Password saved successfully.")

        self.mode = "enrollment"
        self.enroll_count = 0
        self.setup_enrollment_mode()

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

            self.add_card_layout_instructions(card_layout)
            self.add_card_username_input(card_layout)
            self.add_card_password_input(card_layout)
            self.add_card_enrollment_buttons(card_layout)
            self.add_card_enrollment_progress_label(card_layout)

            self.layout.addStretch()
            self.layout.addWidget(card, alignment=Qt.AlignCenter)
            self.layout.addStretch()

            self.data_utility.start()

        except Exception as e:
            QMessageBox.critical(self, "Enrollment Setup Error", str(e))


    def submit_enrollment_sample(self):
        password = self.password_entry.text()
        username = self.username_entry.text()
        try:
            if not username:
                QMessageBox.warning(self, "Enrollment", "Enter a username.")
                self.password_entry.clear()
                self.data_utility.reset()
                return

            self.username = username
            if password != constants.PASSWORD:
                QMessageBox.warning(self, "Enrollment", "Password does not match.")
                self.password_entry.clear()
                self.data_utility.reset(failed=True)
                return

            self.data_utility.set_username(username)
            self.data_utility.extract_features(username)
            filename = "enrollment_features.csv"


            self.data_utility.save_features_csv(
                filename=filename,
                append=self.enroll_append
            )

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
            self.logger.log(self.username, "Completed keyboard data enrollment")
        except Exception as e:
            QMessageBox.critical(self, "Enrollment Error", str(e))

    def load_csv_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Features CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return

        if not os.path.exists(file_path):
            QMessageBox.warning(self, "Load CSV", "Selected file does not exist.")
            return


        try:
            self.data_utility.load_csv_key_features(file_path)
            QMessageBox.information(self, "Load CSV", f"Features successfully loaded from {file_path}.")
            self.logger.log(self.username, f"Successfully loaded features from {file_path}.")
            self.on_mode_changed("Training")
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
            self.data_utility.set_username(username=username)

            preprocessor = DataPreprocessor(
                enrollment_csv=os.path.join(constants.PATH_EXTRACTED, username, "enrollment_features.csv"),
                username=username,
                output_csv=os.path.join(constants.PATH_DATASETS, f"{username}_training.csv")
            )

            trainer = ModelTrainer(
                csv_path=os.path.join(constants.PATH_DATASETS, f"{username}_training.csv"),
                username=username,
                out_dir=constants.PATH_MODELS,
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
        self.logger.log(self.username, f"Model training finished successfully.")
        self.on_mode_changed("Mouse Enrollment")

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

        urow = QHBoxLayout()
        self.username_label = QLabel("Username:")
        self.username_label.setProperty("class", "field-label")
        urow.addWidget(self.username_label, alignment=Qt.AlignVCenter)

        self.username_entry = QLineEdit()
        self.username_entry.setPlaceholderText("Enter your username")
        urow.addWidget(self.username_entry)
        card_layout.addLayout(urow)

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

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self.authenticate_button = QPushButton("Authenticate")
        self.authenticate_button.setProperty("class", "primary")
        self.authenticate_button.clicked.connect(self.authenticate)
        btn_row.addWidget(self.authenticate_button)

        btn_row.addStretch()
        card_layout.addLayout(btn_row)

        self.result_label = QLabel("")
        self.result_label.setProperty("class", "status")
        self.result_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(self.result_label)

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
                self.data_utility.set_username(username=username)
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
                self.logger.log(self.username, f"Authentication successful with distance = {dist:.4f}")
                try: self.switch_to_background_mode()
                except Exception as e:
                    QMessageBox.critical(self, "Background Mode Error", str(e))

                self.data_utility.reset()
                self.password_entry.clear()
            else:
                QMessageBox.critical(self, "Authentication", f"{message}\nDistance = {dist:.4f}")
                self.logger.log(self.username, f"Authentication failed with distance = {dist:.4f}")

                self.data_utility.reset(failed=True)
                self.password_entry.clear()

        except Exception as e:
            QMessageBox.critical(self, "Authentication Error", str(e))

            self.data_utility.reset(failed=True)
            self.password_entry.clear()

    # =================================================================
    #  MOUSE ENROLLMENT AND TRAINING
    # =================================================================
    def setup_mouse_enrollment_mode(self):
        try:
            self.resize(700, 400)
            self.center_on_screen()

            card = self.make_card()
            layout = QVBoxLayout(card)
            layout.setContentsMargins(30, 30, 30, 30)
            layout.setSpacing(16)

            top_row = QHBoxLayout()
            top_row.addWidget(self.create_mode_selector(), alignment=Qt.AlignLeft)

            title = QLabel("Mouse Enrollment")
            title.setProperty("class", "title")
            title.setAlignment(Qt.AlignCenter)

            top_row.addStretch()
            top_row.addWidget(title)
            top_row.addStretch()
            layout.addLayout(top_row)

            instr = QLabel("Mouse data collection will start and run for 3 minutes.")
            instr.setProperty("class", "instr")
            instr.setAlignment(Qt.AlignCenter)
            layout.addWidget(instr)

            self.mouse_status_label = QLabel("Status: Waiting...")
            self.mouse_status_label.setProperty("class", "status")
            self.mouse_status_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.mouse_status_label)

            # Buttons row
            btn_row = QHBoxLayout()
            btn_row.addStretch()

            start_btn = QPushButton("Start Collection")
            start_btn.setProperty("class", "primary")
            start_btn.clicked.connect(self.start_mouse_collection)
            btn_row.addWidget(start_btn)

            load_btn = QPushButton("Load Mouse CSV")
            load_btn.setProperty("class", "secondary")
            load_btn.clicked.connect(self.load_mouse_csv)
            btn_row.addWidget(load_btn)

            btn_row.addStretch()
            layout.addLayout(btn_row)

            self.layout.addStretch()
            self.layout.addWidget(card, alignment=Qt.AlignCenter)
            self.layout.addStretch()

        except Exception as e:
            QMessageBox.critical(self, "Mouse Enrollment Error", str(e))

    # -------------------------------------------------------------------------
    # Mock handler for Load Mouse CSV
    # -------------------------------------------------------------------------
    def load_mouse_csv(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Mouse Features CSV", "", "CSV Files (*.csv)")
            if not file_path:
                return

            # Mock successful loading (replace with real logic later)
            QMessageBox.information(self, "Load Mouse CSV", f"Successfully loaded {file_path}")

            # Switch to Mouse Training mode
            self.on_mode_changed("Mouse Training")

        except Exception as e:
            QMessageBox.critical(self, "Load Mouse CSV Error", str(e))
    def start_mouse_collection(self):
        try:
            self.mouse_status_label.setText("Status: Collecting mouse data...")
            self.mouse_timer = QTimer()
            self.mouse_timer.setSingleShot(True)
            self.mouse_timer.timeout.connect(self.finish_mouse_collection)
            self.mouse_timer.start(1 * 60 * 1000)

            self.data_utility.start_background_collection()

        except Exception as e:
            QMessageBox.critical(self, "Mouse Collection Error", str(e))

    def finish_mouse_collection(self):
        self.data_utility.stop_background_collection()
        self.data_utility.save_mouse_raw_csv(filename="mouse_enrollement.csv")
        self.mouse_status_label.setText("Status: Collection Complete.")
        QMessageBox.information(self, "Mouse Enrollment", "Mouse data collection finished.")

        self.on_mode_changed("Mouse Training")

    def setup_mouse_training_mode(self):
        self.resize(700, 400)
        self.center_on_screen()

        card = self.make_card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(16)

        top_row = QHBoxLayout()
        top_row.addWidget(self.create_mode_selector(), alignment=Qt.AlignLeft)

        title = QLabel("Mouse Training")
        title.setProperty("class", "title")
        title.setAlignment(Qt.AlignCenter)

        top_row.addStretch()
        top_row.addWidget(title)
        top_row.addStretch()
        layout.addLayout(top_row)

        instr = QLabel("This will train the mouse behavioral model based on collected data.")
        instr.setProperty("class", "instr")
        instr.setAlignment(Qt.AlignCenter)
        layout.addWidget(instr)

        self.mouse_training_status = QLabel("Press 'Start Mouse Training' to begin.")
        self.mouse_training_status.setProperty("class", "status")
        self.mouse_training_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.mouse_training_status)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        start_btn = QPushButton("Start Mouse Training")
        start_btn.setProperty("class", "primary")
        start_btn.clicked.connect(self.start_mouse_training)
        btn_row.addWidget(start_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Add card to main layout
        self.layout.addStretch()
        self.layout.addWidget(card, alignment=Qt.AlignCenter)
        self.layout.addStretch()

    def start_mouse_training(self):
        try:
            self.mouse_training_status.setText("Mouse model training in progress... Please wait.")

            enrollment_csv = os.path.join(constants.PATH_EXTRACTED, self.username, "mouse_enrollement.csv")
            dataset_root = os.path.join(constants.PATH_DATASETS, "sapimouse")
            model_out_dir = constants.PATH_MODELS

            trainer = MouseModelTrainer(
                enrollment_csv=enrollment_csv,
                username=self.username,
                dataset_root=dataset_root,
                out_dir=model_out_dir,
                window_size=128,
                step_size=64,
                epochs=15,
                batch_size=32,
                lr=1e-3
            )

            # Train the model
            model_path, auc = trainer.train()
            self.mouse_training_status.setText(f"Training complete.")
            QMessageBox.information(
                self,
                "Mouse Training",
                f"Mouse model training finished successfully.\nModel saved to:\n{model_path}"
            )
            self.on_mode_changed("Authentication")

        except Exception as e:
            QMessageBox.critical(self, "Mouse Training Error", str(e))
            self.mouse_training_status.setText("Training failed.")
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

            elif text == "Mouse Enrollment":
                self.mode = "mouse_enroll"
                self.clear_layout()
                self.setup_mouse_enrollment_mode()

            elif text == "Mouse Training":
                self.mode = "mouse_train"
                self.clear_layout()
                self.setup_mouse_training_mode()

            elif text == "Authentication":
                self.mode = "authentication"
                try:
                    self.authenticator.load_model(
                        os.path.join(constants.PATH_MODELS, f"{self.username}_snn.pt"),
                        username=self.username,
                        training_csv=f"{self.username}_training.csv"
                    )
                except Exception as e:
                    QMessageBox.critical(self, "Model load failed", str(e))
                    return

                self.clear_layout()
                self.setup_authentication_mode()

        except Exception as e:
            QMessageBox.critical(self, "Mode Change Error", str(e))

    # =================================================================
    #  BACKGROUND CONTINUOUS AUTHENTICATION
    # =================================================================
    def switch_to_background_mode(self):
        # ---- UI setup errors ----
        try:
            if hasattr(self, "bg_window") and self.bg_window:
                self.close_background_mode()

            self.bg_window = QWidget()
            self.bg_window.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
            self.bg_window.setAttribute(Qt.WA_TranslucentBackground)
            self.bg_window.setFixedSize(280, 180)

            # Add a QFrame for the visible rounded rectangle
            card = QFrame(self.bg_window)
            card.setObjectName("bg-card-frame")
            card.setGeometry(0, 0, 280, 180)
            card.setStyleSheet("""
                QFrame#bg-card-frame {
                    background-color: #ffffff;
                    border-radius: 20px;
                }
            """)

            layout = QVBoxLayout(card)
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
                screen.x() + screen.width() - self.bg_window.width() - 20,
                screen.y() + screen.height() - self.bg_window.height() - 20
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
                    constants.PATH_MODELS, f"{self.username}_cnn_model.keras"
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
                    f"Behavioral authentication failed. Score: {mean_score}.\nPlease authenticate again."
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
        box.addItems([
            "Enrollment",
            "Training",
            "Mouse Enrollment",
            "Mouse Training",
            "Authentication"
        ])
        box.setCurrentText({
                               "enrollment": "Enrollment",
                               "training": "Training",
                               "mouse_enroll": "Mouse Enrollment",
                               "mouse_train": "Mouse Training",
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
            f"Password must be: {constants.PASSWORD}"
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
        btn_row.addWidget(self.skip_enroll_button)

        btn_row.addStretch()
        card_layout.addLayout(btn_row)

    # =================================================================
    #  SUPERUSER COMPONENTS
    # =================================================================

    def superuser_exists(self):
        """Check if a superuser is already created."""
        return self.user_management_utility.get_superuser() is not None

    class SuperuserDialog(QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Create Superuser")

            self.layout = QVBoxLayout()

            self.username_label = QLabel("Username")
            self.username_input = QLineEdit()
            self.layout.addWidget(self.username_label)
            self.layout.addWidget(self.username_input)

            self.password_label = QLabel("Password")
            self.password_input = QLineEdit()
            self.password_input.setEchoMode(QLineEdit.Password)
            self.layout.addWidget(self.password_label)
            self.layout.addWidget(self.password_input)

            self.submit_btn = QPushButton("Create Superuser")
            self.submit_btn.setProperty("class", "primary")
            self.submit_btn.clicked.connect(self.accept)
            self.layout.addWidget(self.submit_btn)

            self.setLayout(self.layout)

        def get_credentials(self):
            return self.username_input.text(), self.password_input.text()

    def handle_login_superuser(self):
        """
        Step 1: Prompt user to enter a username for superuser login.
        Then redirect to password input page.
        """
        username = self.username_entry.text().strip()
        if not username:
            QMessageBox.warning(self, "Login Superuser", "Please enter a username")
            return

        # Check if superuser exists with this username
        superuser = self.user_management_utility.get_superuser()
        if not superuser or superuser['username'] != username:
            QMessageBox.warning(self, "Login Superuser", "No superuser found with this username")
            return

        # Store username temporarily
        self.username = username

        # Redirect to password page
        self.setup_superuser_password_page()

    def setup_superuser_password_page(self):
        """
        Step 2: Show page to enter superuser password.
        """
        self.clear_layout()
        self.resize(700, 400)
        self.center_on_screen()

        card = self.make_card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(16)

        # Title
        title = QLabel("Superuser Login")
        title.setProperty("class", "title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Enter the superuser password")
        subtitle.setProperty("class", "instr")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        # Password input
        self.superuser_password_entry = QLineEdit()
        self.superuser_password_entry.setEchoMode(QLineEdit.Password)
        self.superuser_password_entry.setPlaceholderText("Enter password")
        layout.addWidget(self.superuser_password_entry)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        login_btn = QPushButton("Login")
        login_btn.setProperty("class", "primary")
        login_btn.clicked.connect(self.authenticate_superuser)
        btn_row.addWidget(login_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setProperty("class", "secondary")
        cancel_btn.clicked.connect(self.setup_landing_page)
        btn_row.addWidget(cancel_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.layout.addStretch()
        self.layout.addWidget(card, alignment=Qt.AlignCenter)
        self.layout.addStretch()

    def authenticate_superuser(self):
        """
        Check if entered password matches the superuser password.
        Redirect to dashboard if successful.
        """
        password = self.superuser_password_entry.text()
        superuser = self.user_management_utility.get_superuser()

        if not superuser:
            QMessageBox.critical(self, "Error", "No superuser found")
            return

        if self.user_management_utility.verify_superuser(self.username, password):
            QMessageBox.information(self, "Superuser Login", "Authentication successful!")
            self.setup_superuser_dashboard()
        else:
            QMessageBox.critical(self, "Superuser Login", "Incorrect password.")
            self.superuser_password_entry.clear()

    def setup_superuser_dashboard(self):
        self.clear_layout()
        self.resize(800, 600)  # slightly taller to fit logs
        self.center_on_screen()

        card = self.make_card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # Title
        title = QLabel(f"Superuser Dashboard - {self.username}")
        title.setProperty("class", "title")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # --- Local user and threshold section ---
        self.local_user = self.user_management_utility.get_local_user_username()
        self.threshold_file = os.path.join(constants.PATH_MODELS, f"{self.local_user}_mouse_threshold.npy")

        self.user_row = QHBoxLayout()
        self.user_row.addWidget(QLabel(f"Username: {self.local_user}"))

        # Threshold label or "Unknown"
        self.threshold_label = QLabel()
        self.update_threshold_label()
        self.user_row.addWidget(self.threshold_label)

        # Edit button
        self.edit_threshold_button = QPushButton("Edit")
        self.edit_threshold_button.clicked.connect(self.enable_threshold_editing)
        self.user_row.addWidget(self.edit_threshold_button)

        layout.addLayout(self.user_row)

        # Save button (hidden until editing)
        self.save_threshold_button = QPushButton("Save")
        self.save_threshold_button.setVisible(False)
        self.save_threshold_button.clicked.connect(self.save_new_threshold)
        layout.addWidget(self.save_threshold_button, alignment=Qt.AlignLeft)

        # Status label
        self.threshold_status_label = QLabel("")
        self.threshold_status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.threshold_status_label)

        # --- Logs view ---
        logs_title = QLabel("Local User Activity Logs:")
        logs_title.setProperty("class", "subtitle")
        layout.addWidget(logs_title)

        self.logs_view = QTextEdit()
        self.logs_view.setReadOnly(True)
        layout.addWidget(self.logs_view)

        # Refresh logs button
        self.refresh_logs_button = QPushButton("Refresh Logs")
        self.refresh_logs_button.clicked.connect(self.update_logs_view)
        layout.addWidget(self.refresh_logs_button, alignment=Qt.AlignLeft)

        # Initialize logger
        self.logger = Logger()  # make sure Logger class is imported

        # Load logs for the first time
        self.update_logs_view()

        self.layout.addStretch()
        self.layout.addWidget(card, alignment=Qt.AlignCenter)
        self.layout.addStretch()

    # --- Helper to refresh logs ---
    def update_logs_view(self):
        logs_text = self.logger.get_logs(self.local_user)
        self.logs_view.setPlainText(logs_text)

    def update_threshold_label(self):
        """Update the label to show current threshold or unknown"""
        if os.path.exists(self.threshold_file):
            try:
                threshold = np.load(self.threshold_file)[0]
                self.threshold_label.setText(f"Threshold: {threshold}")
            except:
                self.threshold_label.setText("Threshold: unknown")
        else:
            self.threshold_label.setText("Threshold: unknown")

    def enable_threshold_editing(self):
        """Turn the threshold label into editable input"""
        # Remove label from layout
        self.user_row.removeWidget(self.threshold_label)
        self.threshold_label.setParent(None)

        # Replace with QLineEdit
        self.threshold_entry = QLineEdit()
        if os.path.exists(self.threshold_file):
            self.threshold_entry.setText(str(np.load(self.threshold_file)[0]))
        self.user_row.insertWidget(1, self.threshold_entry)

        # Show Save button
        self.save_threshold_button.setVisible(True)
        self.edit_threshold_button.setEnabled(False)

    def save_new_threshold(self):
        """Save the new threshold and restore label"""
        try:
            new_threshold = float(self.threshold_entry.text().strip())
        except ValueError:
            QMessageBox.warning(self, "Superuser", "Please enter a valid number")
            return

        # Save to .npy
        np.save(self.threshold_file, np.array([new_threshold]))
        self.threshold_status_label.setText(f"Threshold updated to {new_threshold}")

        # Remove QLineEdit and restore label
        self.user_row.removeWidget(self.threshold_entry)
        self.threshold_entry.setParent(None)
        self.threshold_entry = None

        self.threshold_label = QLabel()
        self.update_threshold_label()
        self.user_row.insertWidget(1, self.threshold_label)

        # Restore buttons
        self.edit_threshold_button.setEnabled(True)
        self.save_threshold_button.setVisible(False)

    def setup_logs_view(self):
        """Add a log viewer to the dashboard"""
        self.logs_view = QTextEdit()
        self.logs_view.setReadOnly(True)
        self.logs_view.setPlaceholderText("User activity logs will appear here...")
        self.layout.addWidget(self.logs_view)

        # Button to refresh logs
        self.refresh_logs_button = QPushButton("Refresh Logs")
        self.refresh_logs_button.clicked.connect(self.load_user_logs(self.local_user))
        self.layout.addWidget(self.refresh_logs_button)

        # Load initial logs
        self.load_user_logs(self.local_user)

    def load_user_logs(self, username):
        self.logs_view.setPlainText(self.logger.get_logs(username))

