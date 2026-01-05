
import sys
from PyQt5.QtWidgets import QApplication, QMessageBox
app = QApplication.instance() or QApplication(sys.argv)
import pytest
from unittest.mock import MagicMock, patch
from src.gui.authentication_window import AuthenticationWindow
import src.constants as constants


constants.PASSWORD = ".tie5Roanl"

@pytest.fixture(autouse=True)
def mock_qmessagebox():
    with patch("PyQt5.QtWidgets.QMessageBox.warning", return_value=None), \
         patch("PyQt5.QtWidgets.QMessageBox.information", return_value=None), \
         patch("PyQt5.QtWidgets.QMessageBox.critical", return_value=None):
        yield

@pytest.fixture(autouse=True)
def mock_qtimer():
    with patch("PyQt5.QtCore.QTimer.singleShot", lambda ms, func: func()):
        yield

@pytest.fixture
def window():
    w = AuthenticationWindow()
    w.data_utility = MagicMock()
    w.username_entry = MagicMock()
    w.username_entry.text.return_value = "ksenia"
    w.password_entry = MagicMock()
    w.password_entry.text.return_value = constants.PASSWORD
    w.progress_label = MagicMock()
    w.schedule_next_enrollment = MagicMock()
    return w

def test_submit_enrollment_success(window):
    window.enroll_count = 0
    window.enroll_target = 40

    window.submit_enrollment_sample()

    window.data_utility.extract_features.assert_called_once_with("ksenia")
    window.data_utility.save_features_csv.assert_called_once()


def test_submit_enrollment_wrong_password(window):
    window.enroll_count = 0
    window.enroll_target = 40
    window.password_entry.text.return_value = "wrong_password"

    with patch.object(QMessageBox, "warning", return_value=None) as mock_warning:
        window.submit_enrollment_sample()

    window.data_utility.extract_features.assert_not_called()
    window.data_utility.save_features_csv.assert_not_called()
    window.data_utility.reset.assert_called_with(failed=True)
    mock_warning.assert_called_once_with(window, "Enrollment", "Password does not match.")


def test_submit_enrollment_empty_username(window):
    window.username_entry.text.return_value = ""

    window.submit_enrollment_sample()

    window.data_utility.extract_features.assert_not_called()
    window.data_utility.save_features_csv.assert_not_called()
    window.data_utility.reset.assert_called()


def test_save_password_and_continue_success(window):
    window.new_password_entry = MagicMock()
    window.confirm_password_entry = MagicMock()
    window.new_password_entry.text.return_value = "newpass"
    window.confirm_password_entry.text.return_value = "newpass"
    window.setup_enrollment_mode = MagicMock()

    window.save_password_and_continue()

    assert window.mode == "enrollment"
    assert window.enroll_count == 0
    assert constants.PASSWORD == "newpass"
    window.setup_enrollment_mode.assert_called_once()


def test_save_password_and_continue_mismatch(window):
    window.new_password_entry = MagicMock()
    window.confirm_password_entry = MagicMock()
    window.new_password_entry.text.return_value = "pass1"
    window.confirm_password_entry.text.return_value = "pass2"
    window.setup_enrollment_mode = MagicMock()

    window.save_password_and_continue()

    assert window.mode != "enrollment"
    window.setup_enrollment_mode.assert_not_called()


def test_handle_login_empty_username(window):
    window.username_entry.text.return_value = ""
    with patch("PyQt5.QtWidgets.QMessageBox.warning") as mock_warn:
        window.handle_login()
        mock_warn.assert_called_once()


def test_handle_login_model_missing(window):
    window.username_entry.text.return_value = "ksenia"
    with patch("os.path.exists", return_value=False), \
         patch("PyQt5.QtWidgets.QMessageBox.information") as mock_info:
        window.handle_login()
        mock_info.assert_called_once()


def test_authenticate_success(window):
    window.username_entry.text.return_value = "ksenia"
    window.password_entry.text.return_value = constants.PASSWORD
    window.data_utility.feature_extractor = MagicMock()
    window.data_utility.feature_extractor.key_features = MagicMock()
    window.data_utility.feature_extractor.key_features.data = "features"
    window.data_utility.extract_features = MagicMock()
    window.data_utility.save_features_csv = MagicMock()
    window.authenticator.authenticate = MagicMock(return_value=(True, 0.1, "Welcome"))
    window.data_utility.reset = MagicMock()
    with patch("PyQt5.QtWidgets.QMessageBox.information") as mock_info:
        window.authenticate()
        mock_info.assert_called_once()


def test_authenticate_fail(window):
    window.username_entry.text.return_value = "ksenia"
    window.password_entry.text.return_value = constants.PASSWORD
    window.data_utility.feature_extractor = MagicMock()
    window.data_utility.feature_extractor.key_features = MagicMock()
    window.data_utility.feature_extractor.key_features.data = "features"
    window.data_utility.extract_features = MagicMock()
    window.data_utility.save_features_csv = MagicMock()
    window.authenticator.authenticate = MagicMock(return_value=(False, 0.5, "Denied"))
    window.data_utility.reset = MagicMock()
    with patch("PyQt5.QtWidgets.QMessageBox.critical") as mock_crit:
        window.authenticate()
        mock_crit.assert_called_once()


def test_on_auth_result_accepted(window):
    window.bg_manager = MagicMock()
    window.bg_status_label = MagicMock()
    window.bg_result_label = MagicMock()
    window.data_utility.clear_mouse_data = MagicMock()
    with patch("PyQt5.QtCore.QTimer.singleShot") as mock_timer:
        window.on_auth_result(True, 0.42)
        window.bg_status_label.setText.assert_called_with("● Authenticated")
        window.bg_result_label.setText.assert_called_with("✓ Accepted (0.42)")
        mock_timer.assert_called_once()


def test_on_auth_result_rejected(window):
    window.bg_manager = MagicMock()
    window.bg_result_label = MagicMock()
    window.password_entry = MagicMock()
    with patch("PyQt5.QtWidgets.QMessageBox.warning") as mock_warn, \
         patch.object(window, "close_background_mode") as mock_close:
        window.on_auth_result(False, 0.75)
        window.bg_result_label.setText.assert_called_with("✗ Rejected (0.75)")
        mock_warn.assert_called_once()
        mock_close.assert_called_once()
