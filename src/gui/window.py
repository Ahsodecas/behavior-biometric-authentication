from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QMessageBox
)
from PyQt5.QtCore import Qt, QEvent, QTimer
from src.utils.data_collector import DataCollector


class AuthWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Authentication App")
        self.mode = "authentication"
        self.data_collector = DataCollector()
        self.setup_authentication_mode()

    def setup_authentication_mode(self):
        """Setup full-screen authentication mode"""
        self.resize(600, 400)
        self.center_on_screen()

        # Layout
        self.layout = QVBoxLayout()

        # Username label + entry
        self.username_label = QLabel("Username:")
        self.username_label.setStyleSheet("font-size: 14pt;")
        self.layout.addWidget(self.username_label, alignment=Qt.AlignCenter)

        self.username_entry = QLineEdit()
        self.username_entry.setPlaceholderText("Enter your username")
        self.username_entry.setStyleSheet("font-size: 12pt; padding: 5px;")
        self.layout.addWidget(self.username_entry, alignment=Qt.AlignCenter)

        # Password label + entry
        self.password_label = QLabel("Password:")
        self.password_label.setStyleSheet("font-size: 14pt;")
        self.layout.addWidget(self.password_label, alignment=Qt.AlignCenter)

        self.password_entry = QLineEdit()
        self.password_entry.setEchoMode(QLineEdit.Password)
        self.password_entry.setPlaceholderText("Enter your password")
        self.password_entry.setStyleSheet("font-size: 12pt; padding: 5px;")
        self.layout.addWidget(self.password_entry, alignment=Qt.AlignCenter)

        # Install event filters to capture key press/release
        self.password_entry.installEventFilter(self)

        # Start session for keystroke collection
        self.data_collector.start_session()

        # Authenticate button
        self.authenticate_button = QPushButton("Authenticate")
        self.authenticate_button.setStyleSheet(
            "font-size: 12pt; padding: 10px 20px;"
        )
        self.authenticate_button.clicked.connect(self.authenticate)
        self.layout.addWidget(self.authenticate_button, alignment=Qt.AlignCenter)

        # Result label
        self.result_label = QLabel("")
        self.result_label.setStyleSheet("font-size: 12pt;")
        self.layout.addWidget(self.result_label, alignment=Qt.AlignCenter)

        self.setLayout(self.layout)

    def eventFilter(self, obj, event):
        """Capture keystroke events for password field"""
        if obj == self.password_entry:
            if event.type() == QEvent.KeyPress:
                self.data_collector.collect_key_event(event, "press")
            elif event.type() == QEvent.KeyRelease:
                self.data_collector.collect_key_event(event, "release")
        return super().eventFilter(obj, event)


    def authenticate(self):
        username = self.username_entry.text()
        password = self.password_entry.text()

        if not username or not password:
            QMessageBox.critical(self, "Authentication", "Please enter both username and password.")
            self.result_label.setText("")
            return

        if password != "TestPassword123":
            QMessageBox.critical(self, "Authentication", "Password incorrect.")
            self.result_label.setText("Authentication failed.")
            return

        self.data_collector.username = username
        self.data_collector.extract_features()

        # Simulate authentication result
        is_auth = True
        score = 0.97  # dummy score

        if is_auth:
            self.data_collector.save_session_csv()
            QMessageBox.information(self, "Authentication",
                                    f"Authentication successful with score: {score:.2f}!")
            self.result_label.setText(f"Authenticated as: {username}")
            QTimer.singleShot(1000, self.switch_to_background_mode)
        else:
            QMessageBox.critical(self, "Authentication",
                                 f"Authentication failed with score: {score:.2f}.")
            self.result_label.setText("Authentication failed.")

    def switch_to_background_mode(self):
        """Switch to background monitoring mode"""
        self.mode = "background"
        self.clear_layout(self.layout)
        self.setup_background_mode()

    def setup_background_mode(self):
        """Setup small background mode window"""
        screen = QApplication.primaryScreen().availableGeometry()
        window_width, window_height = 250, 100
        x = screen.width() - window_width - 20
        y = screen.height() - window_height - 40
        self.setGeometry(x, y, window_width, window_height)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout()

        status_label = QLabel("Status: Authenticated")
        status_label.setStyleSheet("font-size: 10pt; color: green;")
        layout.addWidget(status_label, alignment=Qt.AlignCenter)

        auth_status = QLabel("✓ Continuous Monitoring Active")
        auth_status.setStyleSheet("font-size: 9pt;")
        layout.addWidget(auth_status, alignment=Qt.AlignCenter)

        exit_button = QPushButton("Exit")
        exit_button.setStyleSheet("font-size: 9pt; padding: 5px;")
        exit_button.clicked.connect(self.close)
        layout.addWidget(exit_button, alignment=Qt.AlignCenter)

        self.setLayout(layout)
        self.show()

    def clear_layout(self, layout):
        """Helper to clear all widgets from a layout"""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def center_on_screen(self):
        """Center the window on screen"""
        frame_geo = self.frameGeometry()
        screen = QApplication.primaryScreen().availableGeometry().center()
        frame_geo.moveCenter(screen)
        self.move(frame_geo.topLeft())
