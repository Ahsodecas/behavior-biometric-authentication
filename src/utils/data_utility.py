from src.utils.data_collector import DataCollector
from src.utils.feature_extractor import FeatureExtractor
from src.utils.synthetic_features_generator import SyntheticFeaturesGenerator

class DataUtility:
    def __init__(self):
        self.data_collector = DataCollector()
        self.feature_extractor = FeatureExtractor()
        self.synthetic_features_generator = SyntheticFeaturesGenerator()

    def start(self):
        self.data_collector.start_session()

    def feed_key_event(self, event, event_type):
        self.data_collector.collect_key_event(event, event_type)

    def extract_features(self, username):
        self.data_collector.username = username
        self.feature_extractor.username = username
        self.feature_extractor.raw_key_data = self.data_collector.data
        self.feature_extractor.extract_key_features()

    def generate_synthetic_features(self, username):
        self.synthetic_features_generator.username = username
        self.synthetic_features_generator.genuine_features = self.feature_extractor.prepocess_features_for_synthesis()
        # add generation function and some parameters for it
        self.synthetic_features_generator.generate()

    def save_raw_csv(self, filename=None):
        self.data_collector.save_key_raw_csv(filename)

    def save_features_csv(self, filename=None, append=False):
        self.feature_extractor.save_key_features_csv(filename, append)

    def reset(self):
        self.data_collector.clear_for_next_rep()
        self.feature_extractor.clear_for_next_rep()