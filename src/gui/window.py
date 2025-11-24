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
from src.utils.data_collector import DataCollector

model_path = "models/snn_final.pt"

# ----------------- App-wide CSS (tweakable) -----------------


class AuthWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Authentication App")

        self.mode = "enrollment"
        self.enroll_target = 40
        self.enroll_count = 0
        self.password_fixed = ".tie5Roanl"
        self.enroll_filename = None

        self.data_collector = DataCollector()
        self.setup_enrollment_mode()

        self.model = None
        self.scaler: StandardScaler = None
        self.feature_cols = None
        self.ref_sample = None
        self.templates = {}
        self.threshold = 0.2
        self.device = None

    # ---------------- UI BUILD HELPERS ----------------
    def center_on_screen(self):
        frame = self.frameGeometry()
        screen = QApplication.primaryScreen().availableGeometry().center()
        frame.moveCenter(screen)
        self.move(frame.topLeft())

    def make_card(self):
        """Return a larger, modern card."""
        card = QFrame()
        card.setObjectName("card")
        card.setMinimumWidth(600)
        card.setMinimumHeight(400)
        return card

    def setup_enrollment_mode(self):
        if hasattr(self, "layout") and self.layout:
            self.clear_layout()

        self.resize(900, 650)
        self.center_on_screen()

        self.layout = QVBoxLayout(self)
        self.layout.setAlignment(Qt.AlignCenter)
        self.layout.setSpacing(20)

        card = self.make_card()
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

        # Instruction
        instr = QLabel(
            f"Please type your password exactly {self.enroll_target} times.\nPassword must be: {self.password_fixed}")
        instr.setProperty("class", "instr")
        instr.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(instr)

        # Username input
        username_row = QHBoxLayout()
        username_label = QLabel("Username:")
        username_label.setProperty("class", "field-label")
        username_row.addWidget(username_label, alignment=Qt.AlignVCenter)
        self.username_entry = QLineEdit()
        self.username_entry.setPlaceholderText("Choose a username")
        username_row.addWidget(self.username_entry)
        card_layout.addLayout(username_row)

        # Password input
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

        # Buttons
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

        # Progress label
        self.progress_label = QLabel(f"Samples collected: 0 / {self.enroll_target}")
        self.progress_label.setProperty("class", "status")
        self.progress_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(self.progress_label)

        self.layout.addStretch()
        self.layout.addWidget(card, alignment=Qt.AlignCenter)
        self.layout.addStretch()
        self.setLayout(self.layout)

        self.data_collector.start_session()
        self.enroll_filename = f"enrollment_features_ksenia.csv"

    def skip_enrollment(self):
        # Let user select existing enrollment CSV
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Enrollment CSV", "", "CSV Files (*.csv);;All Files (*)", options=options
        )
        if file_name:
            self.enroll_filename = file_name
            QMessageBox.information(self, "Enrollment Skipped", f"Loaded CSV: {file_name}\nProceeding to training.")
            self.switch_to_training_mode()

    def submit_enrollment_sample(self):
        username = self.username_entry.text()
        password = self.password_entry.text()

        if not username:
            QMessageBox.warning(self, "Enrollment", "Enter a username.")
            self.password_entry.clear()
            self.data_collector.clear_for_next_rep(failed=True)
            return

        if password != self.password_fixed:
            QMessageBox.warning(self, "Enrollment", "Password does not match the required one.")
            self.password_entry.clear()
            self.data_collector.clear_for_next_rep(failed=True)
            return

        self.data_collector.username = username
        self.data_collector.extract_features()
        self.data_collector.save_features_csv(filename=self.enroll_filename, append=True)
        self.enroll_count += 1
        self.progress_label.setText(f"Samples collected: {self.enroll_count} / {self.enroll_target}")

        self.password_entry.clear()
        self.data_collector.clear_for_next_rep()

        if self.enroll_count >= self.enroll_target:
            QMessageBox.information(
                self, "Enrollment Complete",
                f"Collected {self.enroll_target} samples.\nSwitching to training phase."
            )
            self.switch_to_training_mode()

    # ---------------------------------------------------------
    # -------------------- TRAINING MODE ----------------------
    # ---------------------------------------------------------
    def setup_training_mode(self):
        self.resize(600, 400)
        self.center_on_screen()

        self.clear_layout()  # remove old widgets

        self.layout.addWidget(QLabel("Mode:"), alignment=Qt.AlignLeft)
        self.layout.addWidget(self.create_mode_selector(), alignment=Qt.AlignLeft)

        title = QLabel("Training SNN Model")
        title.setStyleSheet("font-size: 18pt; font-weight: bold;")
        self.layout.addWidget(title, alignment=Qt.AlignCenter)

        self.train_progress_label = QLabel("Training progress: 0 / 0")
        self.train_progress_label.setStyleSheet("font-size: 12pt;")
        self.layout.addWidget(self.train_progress_label, alignment=Qt.AlignCenter)

        self.start_training_button = QPushButton("Start Training")
        self.start_training_button.setStyleSheet("font-size: 12pt; padding: 10px 20px;")
        self.start_training_button.clicked.connect(self.start_training)
        self.layout.addWidget(self.start_training_button, alignment=Qt.AlignCenter)

    def start_training(self):
        # Combine enrollment CSV + another dataset if needed
        csv_files = [
            self.enroll_filename,
            os.path.join("datasets", "DSL-StrongPasswordData.csv")
        ]  # adjust path as needed
        out_dir = "checkpoints"
        self.start_training_button.setEnabled(False)
        self.train_thread = self.TrainingThread(csv_files, out_dir, epochs=10)
        self.train_thread.progress.connect(self.update_training_progress)
        self.train_thread.finished.connect(self.training_finished)
        self.train_thread.start()

    def update_training_progress(self, epoch, total_epochs):
        self.train_progress_label.setText(f"Training progress: {epoch} / {total_epochs}")

    def training_finished(self, model_path):
        self.train_progress_label.setText("Training complete!")
        QMessageBox.information(self, "Training", f"Model saved to {model_path}")
        # After training, optionally switch to authentication mode
        self.switch_to_authentication_mode()

    # ---------------------------------------------------------
    # ------------------- VERIFICATION MODE -------------------
    # ---------------------------------------------------------
    def setup_authentication_mode(self):
        self.clear_layout()
        self.resize(760, 480)
        self.center_on_screen()

        self.layout.setAlignment(Qt.AlignCenter)

        card = self.make_card()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 16, 18, 16)
        card_layout.setSpacing(12)

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

        # Username
        urow = QHBoxLayout()
        self.username_label = QLabel("Username:")
        self.username_label.setProperty("class", "field-label")
        urow.addWidget(self.username_label, alignment=Qt.AlignVCenter)
        self.username_entry = QLineEdit()
        self.username_entry.setPlaceholderText("Enter your username")
        urow.addWidget(self.username_entry)
        card_layout.addLayout(urow)

        # Password
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

        self.layout.addStretch()
        self.layout.addWidget(card, alignment=Qt.AlignCenter)
        self.layout.addStretch()
        self.setLayout(self.layout)

        # start data collector session for auth mode
        self.data_collector.start_session()

    def load_snn_model(self, ckpt_path):
        # Choose device (you can change to cuda if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device set to {self.device}")

        # check path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Temporary dataset to infer input dim and obtain scaler / feature columns
        tmp_dataset = CMUDatasetTriplet("datasets/ksenia_training_2.csv")
        input_dim = tmp_dataset.X.shape[1]
        print(f"input dim: {input_dim}")

        # Save scaler/feature_cols/ref_sample for later normalization and reference embedding
        self.scaler = tmp_dataset.scaler
        self.feature_cols = tmp_dataset.feature_cols
        # Save a reference normalized sample (first row) to compare against
        if tmp_dataset.X.shape[0] < 40:
            raise ValueError("Dataset must contain at least 40 samples to compute mean reference sample.")
        self.ref_sample = tmp_dataset.X[:40].mean(axis=0).astype(np.float32)

        print(f"Loaded ref_sample for user {tmp_dataset.y[0]}: {self.ref_sample}")

        # build model, load weights, set to device and eval
        model = TripletSNN(input_dim=input_dim)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(ckpt["model_state"])
        model.to(self.device)
        model.eval()

        self.model = model
        print("Model loaded successfully.")

    # ---------------------------------------------------------
    # -------------------- AUTHENTICATION ----------------------
    # ---------------------------------------------------------
    def embed_sample(self, sample_vector):
        if self.model is None:
            raise RuntimeError("Model not loaded! Call load_snn_model first.")
        self.model.eval()

        # convert to 1D numpy / list if it's a single-row 2D array
        arr = np.asarray(sample_vector).reshape(-1).astype(np.float32)
        x = torch.tensor(arr).float().unsqueeze(0).unsqueeze(0)  # (1,1,F)
        lengths = torch.tensor([1], dtype=torch.long)
        with torch.no_grad():
            emb = self.model.subnet(x.to(self.device), lengths.to(self.device))
        return emb.cpu().numpy()[0]

    def verify_user(self, normalized_sample_vector):
        """
        normalized_sample_vector: 1D numpy array already transformed with the same scaler used for training
        """
        if self.ref_sample is None:
            raise RuntimeError("Reference sample not available. Ensure load_snn_model() was called.")

        print("Embedding reference...")
        ref_emb = self.embed_sample(self.ref_sample)

        print("Embedding sample...")
        sample_emb = self.embed_sample(normalized_sample_vector)

        dist = np.linalg.norm(sample_emb - ref_emb)
        is_match = dist < self.threshold
        return is_match, dist

    def authenticate(self):
        username = self.username_entry.text()
        password = self.password_entry.text()

        if not username or not password:
            QMessageBox.warning(self, "Authentication", "Fill all fields.")
            return

        if password != self.password_fixed:
            QMessageBox.critical(self, "Authentication", "Password incorrect.")
            return

        # Ensure model & scaler are loaded
        if self.model is None or self.scaler is None or self.feature_cols is None:
            QMessageBox.critical(self, "Authentication", "Model or scaler not loaded. Switch to Authentication mode to load model.")
            return

        # Collect features from typing
        self.data_collector.username = username
        self.data_collector.extract_features()
        self.data_collector.save_features_csv()
        print("Features collected:", self.data_collector.features)

        try:
            raw = np.array([self.data_collector.features[col] for col in self.feature_cols], dtype=np.float32)
        except KeyError as e:
            QMessageBox.critical(self, "Authentication", f"Missing feature from DataCollector: {e}")
            self.password_entry.clear()
            return
        except Exception as e:
            QMessageBox.critical(self, "Authentication", f"Error reading features: {e}")
            self.password_entry.clear()
            return

        # Normalize using the SAME scaler used for training
        try:
            normalized = self.scaler.transform(raw.reshape(1, -1)).astype(np.float32)[0]
        except Exception as e:
            QMessageBox.critical(self, "Authentication", f"Scaler transform failed: {e}")
            return

        print("Normalized sample:", normalized)

        is_auth, dist = self.verify_user(normalized)

        if is_auth:
            QMessageBox.information(
                self,
                "Authentication",
                f"Authenticated!\nDistance = {dist:.4f}\nThreshold = {self.threshold}"
            )
            self.switch_to_background_mode()

        else:
            QMessageBox.critical(
                self,
                "Authentication",
                f"Rejected!\nDistance = {dist:.4f}\nThreshold = {self.threshold}"
            )

    # ---------------------------------------------------------
    # --------------- EVENT FILTER FOR KEYSTROKES --------------
    # ---------------------------------------------------------
    def eventFilter(self, obj, event):
        if obj == self.password_entry:
            if event.type() == QEvent.KeyPress:
                self.data_collector.collect_key_event(event, "press")
            elif event.type() == QEvent.KeyRelease:
                self.data_collector.collect_key_event(event, "release")

        return super().eventFilter(obj, event)

    # ---------------------------------------------------------
    # -------------------- MODE SWITCHING ---------------------
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    def setup_background_mode(self):
        """Small floating window."""
        screen = QApplication.primaryScreen().availableGeometry()
        w, h = 250, 100
        x = screen.width() - w - 20
        y = screen.height() - h - 40
        self.setGeometry(x, y, w, h)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout()
        status = QLabel("Status: Authenticated")
        status.setStyleSheet("font-size: 10pt; color: green;")
        layout.addWidget(status, alignment=Qt.AlignCenter)
        ok = QLabel("✓ Continuous Monitoring Active")
        ok.setStyleSheet("font-size: 9pt;")
        layout.addWidget(ok, alignment=Qt.AlignCenter)

        exit_btn = QPushButton("Exit")
        exit_btn.clicked.connect(self.close)
        layout.addWidget(exit_btn, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    # ---------------------------------------------------------
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

    def clear_layout(self):
        # Remove all widgets from layout safely
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
