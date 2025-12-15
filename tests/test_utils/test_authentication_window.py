
import sys
from PyQt5.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)
import pytest
from unittest.mock import MagicMock, patch
from src.gui.authentication_window import AuthenticationWindow
import src.constants as constants

constants.PASSWORD = ".tie5Roanl"

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



def test_submit_enrollment_wrong_password(window, mocker):
    import src.constants as constants

    constants.PASSWORD = "CORRECT_PASSWORD"

    window.data_utility.reset = mocker.MagicMock()

    window.submit_enrollment_sample()

    window.data_utility.extract_features.assert_not_called()
    window.data_utility.save_features_csv.assert_not_called()
    window.data_utility.reset.assert_called_once_with(failed=True)

def test_load_csv_missing_file(window, mocker):
    mocker.patch(
        "src.gui.authentication_window.QFileDialog.getOpenFileName",
        return_value=("missing.csv", "CSV Files (*.csv)")
    )
    mocker.patch("os.path.exists", return_value=False)

    window.load_csv_data()

    window.data_utility.load_csv_key_features.assert_not_called()

