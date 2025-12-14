from src.utils.data_collector import DataCollector
from src.utils.feature_extractor import FeatureExtractor
from src.utils.keystroke_dataset_reader import KeystrokeDatasetReader
from src.utils.synthetic_features_generator import SyntheticFeaturesGenerator

class DataUtility:
    def __init__(self):
        self.data_collector = DataCollector()
        self.feature_extractor = FeatureExtractor()
        self.synthetic_features_generator_genuine_user = SyntheticFeaturesGenerator()
        self.synthetic_features_generator_imposter_users = SyntheticFeaturesGenerator()
        self.synthetic_features_generator_other_users = SyntheticFeaturesGenerator()
        self.keystrokeDatasetReader = KeystrokeDatasetReader()

    def start(self):
        self.data_collector.start_session()

    def feed_key_event(self, event, event_type):
        self.data_collector.collect_key_event(event, event_type)

    def extract_features(self, username):
        self.data_collector.username = username
        self.feature_extractor.username = username
        self.feature_extractor.raw_key_data = self.data_collector.data
        self.feature_extractor.extract_key_features()

    def generate_synthetic_features(self, username, filename, repetitions=10):
        self.feature_extractor.username = username
        self.synthetic_features_generator_genuine_user.username = username
        self.synthetic_features_generator_genuine_user.genuine_features = self.feature_extractor.preprocess_features_for_synthesis()
        # add generation function and some parameters for it, change the loop to internal generator logic
        for _ in range(0, repetitions):
            generated_features = self.synthetic_features_generator_genuine_user.generate()
            print("GENERATED FEATURES in DATA UTILITY:")
            print(generated_features)
            self.feature_extractor.key_features.update(metadata=self.feature_extractor.key_features.metadata, features=generated_features)
            print("GENERATED FEATURES in FEATURE EXTRACTOR:")
            print(self.feature_extractor.key_features.features)
            self.save_features_csv(filename=filename,append=True)



    def generate_synthetic_features_other_users(self, username, filename, repetitions=1):
        dataset_dir = "C:\\Users\\anast\\src\\repos\\behavior-biometric-authentication\\datasets\\DATASET"
        self.keystrokeDatasetReader.generate_features_file()
        hold_features, flight_features, ud_features = self.keystrokeDatasetReader.generate_features_file()
        self.synthetic_features_generator_imposter_users.generate_other_users()
        features = {}
        self.synthetic_features_generator.username = username
        self.synthetic_features_generator.genuine_features = dataset_features
        # add generation function and some parameters for it, change the loop to internal generator logic
        for _ in range(0, repetitions):
            generated_features = self.synthetic_features_generator.generate_other_users()
            print("GENERATED FEATURES in DATA UTILITY:")
            print(generated_features)
            features.update(generated_features)
            print("GENERATED FEATURES in FEATURE DICTIONARY:")
            print(features)
        # save it to DSL-StrongPasswordData.csv

    def save_raw_csv(self, filename=None):
        self.data_collector.save_key_raw_csv(filename)

    def save_features_csv(self, filename=None, append=False):
        self.feature_extractor.save_key_features_csv(filename, append)

    def reset(self):
        self.data_collector.clear_for_next_rep()
        self.feature_extractor.clear_for_next_rep()


testDataUtility = DataUtility()
testDataUtility.generate_synthetic_features_other_users(username="test", filename="synthetic_features_other_users.csv", repetitions=1)
