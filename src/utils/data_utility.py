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
        self.data_collector.save_key_raw_csv(filename="raw.csv")
        self.feature_extractor.extract_key_features()

    def generate_synthetic_features(self, username, filename, repetitions=10):
        self.feature_extractor.username = username
        self.synthetic_features_generator.username = username
        self.synthetic_features_generator.genuine_features = self.feature_extractor.prepocess_features_for_synthesis()
        #print("GENUINE FEATURES: ")
        #print(self.synthetic_features_generator.genuine_features)
        # add generation function and some parameters for it, change the loop to internal generator logic
        for i in range(0, repetitions):
            generated_features = self.synthetic_features_generator.generate()
            #print("GENERATED FEATURES in DATA UTILITY:")
            #print(generated_features)
            new_metadata = {"subject" : username,
                            "sessionIndex": -1,
                            "generated" : 1,
                            "rep": i}
            # new_metadata["subject"] = username
            # new_metadata["sessionIndex"] = -1
            # new_metadata["generated"] = 1
            # new_metadata["rep"] = i
            self.feature_extractor.key_features.update(metadata=new_metadata, features=generated_features)
            #print("GENERATED FEATURES in FEATURE EXTRACTOR:")
            #print(self.feature_extractor.key_features.features)
            self.save_features_csv(filename=filename,append=True)

    def save_raw_csv(self, filename=None):
        self.data_collector.save_key_raw_csv(filename)

    def save_features_csv(self, filename=None, append=False):
        self.feature_extractor.save_key_features_csv(filename, append)

    def reset(self):
        self.data_collector.clear_for_next_rep()
        self.feature_extractor.clear_for_next_rep()