from src.utils.data_collector import DataCollector
from src.utils.feature_extractor import FeatureExtractor
from src.ml.keystroke_dataset_reader import KeystrokeDatasetReader
from src.utils.synthetic_features_generator import SyntheticFeaturesGenerator

class DataUtility:
    def __init__(self):
        self.data_collector = DataCollector()
        self.feature_extractor = FeatureExtractor()
        self.synthetic_features_generator_genuine_user = SyntheticFeaturesGenerator()
        self.synthetic_features_generator_imposter_users = SyntheticFeaturesGenerator(username="imposter")
        self.keystrokeDatasetReader = KeystrokeDatasetReader()

    def start(self):
        self.data_collector.start_session()

    def feed_key_event(self, event, event_type):
        self.data_collector.collect_key_event(event, event_type)

    def extract_features(self, username):
        self.set_username(username=username)
        self.feature_extractor.raw_key_data = self.data_collector.data
        self.data_collector.save_key_raw_csv(filename="raw.csv")
        self.feature_extractor.extract_key_features()

    def generate_synthetic_features(self, filename, repetitions=10):
        self.synthetic_features_generator_genuine_user.genuine_features = self.feature_extractor.preprocess_features_for_synthesis()
        print("GENUINE FEATURES: ")
        print(self.synthetic_features_generator_genuine_user.genuine_features)
        generated_features = self.synthetic_features_generator_genuine_user.generate(repetitions=repetitions)

        self.save_generated_features_csv(generated_features=generated_features, filename=filename, append=True, repetitions=repetitions)


    def generate_synthetic_features_imposter_users(self, filename, repetitions=1):
        hold_features, dd_features, ud_features = self.keystrokeDatasetReader.load_key_dataset()
        generated_features = self.synthetic_features_generator_imposter_users.generate(hold_features=hold_features, dd_features=dd_features, ud_features=ud_features, repetitions=repetitions)
        print("GENERATED IMPOSTER FEATURES, REPETITIONS: " + str(repetitions))
        print(generated_features)
        self.save_generated_features_csv(generated_features=generated_features, filename=filename, append=True, repetitions=repetitions)


    def save_generated_features_csv(self, generated_features: list[list[tuple[str, float]]], filename: str, append: bool = True, repetitions: int = 1):

        for i in range(0, repetitions):
            new_metadata = {"subject" : self.feature_extractor.username,
                            "sessionIndex": -1,
                            "generated" : 1,
                            "rep": i}

            self.feature_extractor.key_features.update(metadata=new_metadata, features=dict(generated_features[i]))
            print("GENERATED FEATURES in FEATURE EXTRACTOR:")
            print(self.feature_extractor.key_features.features)
            self.save_features_csv(filename=filename, append=append)

    def load_csv_key_features(self, filename: str) -> str:
        username = self.feature_extractor.load_csv_key_features(filename)
        self.set_username(username=username)
        return username

    def set_username(self, username: str):
        self.data_collector.set_username(username=username)
        self.feature_extractor.set_username(username=username)
        self.synthetic_features_generator_genuine_user.set_username(username=username)

    def save_raw_csv(self, filename=None):
        self.data_collector.save_key_raw_csv(filename)

    def save_features_csv(self, filename=None, append=False):
        self.feature_extractor.save_key_features_csv(filename, append)

    def reset(self):
        self.data_collector.clear_for_next_rep()
        self.feature_extractor.clear_for_next_rep()

#
# testDataUtility = DataUtility()
# testDataUtility.generate_synthetic_features_other_users(username="test", filename="synthetic_features_other_users.csv", repetitions=1)
