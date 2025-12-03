import pytest
import sys
from src.gui.authentication_window import AuthenticationWindow
from PyQt5.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)

@pytest.fixture
def window(mocker):
    w = AuthenticationWindow()

    w.data_utility = mocker.MagicMock()

    w.username_entry.setText("ksenia")
    w.password_entry.setText(".tie5Roanl")

    return w


def test_submit_enrollment_success(window):
    window.enroll_count = 0
    window.enroll_target = 40

    window.submit_enrollment_sample()

    window.data_utility.extract_features.assert_called_once_with("ksenia")
    window.data_utility.save_features_csv.assert_called_once()

    assert window.enroll_count == 1
    assert "1 / 40" in window.progress_label.text()


def test_submit_enrollment_missing_username(window):
    window.username_entry.setText("")

    window.submit_enrollment_sample()

    window.data_utility.extract_features.assert_not_called()
    window.data_utility.save_features_csv.assert_not_called()
    window.data_utility.reset.assert_called_once()


def test_submit_enrollment_wrong_password(window):
    window.password_entry.setText("incorrect")

    window.submit_enrollment_sample()

    window.data_utility.extract_features.assert_not_called()
    window.data_utility.save_features_csv.assert_not_called()
    window.data_utility.reset.assert_called_once()


def test_load_csv_data_success(window, mocker):
    mocker.patch(
        "src.gui.authentication_window.QFileDialog.getOpenFileName",
        return_value=("test_load_csv.csv", "CSV Files")
    )

    mocker.patch("os.path.exists", return_value=True)

    window.data_utility.feature_extractor = mocker.MagicMock()
    window.data_utility.generate_synthetic_features = mocker.MagicMock()

    window.load_csv_data()

    window.data_utility.feature_extractor.key_features.load_csv_features.assert_called_once()
    window.data_utility.generate_synthetic_features.assert_called_once()


def test_load_csv_missing_file(window, mocker):

    mocker.patch(
        "src.gui.authentication_window.QFileDialog.getOpenFileName",
        return_value=("test_load_missing_csv.csv", "CSV Files")
    )
    mocker.patch("os.path.exists", return_value=False)

    window.load_csv_data()

    if hasattr(window.data_utility, "feature_extractor"):
        loader = window.data_utility.feature_extractor.key_features.load_csv_features
        loader.assert_not_called()


# def test_start_model_training(window, mocker):
#     model_trainer = mocker.patch("src.ml.model_trainer.ModelTrainer")
#     data_processor = mocker.patch("src.ml.data_preprocessor.DataPreprocessor")
#     training_worker = mocker.patch("src.ml.training_worker.TrainingWorker")
#
#     window.start_model_training()
#
#     data_processor.assert_called_once()
#     model_trainer.assert_called_once()
#     training_worker.assert_called_once()
#
#     training_worker.return_value.start.assert_called_once()
