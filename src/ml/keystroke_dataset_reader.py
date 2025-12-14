import os
import csv

class KeystrokeDatasetReader:
    def __init__(self):
        self.features = []
        self.dataset_dir = "/datasets/DATASET"
        self.filename = "DSL-StrongPasswordData.csv"

    VK_TO_TOKEN = {
        **{vk: chr(vk) for vk in range(65, 91)},  # A–Z
        **{vk: chr(vk) for vk in range(48, 58)},  # 0–9

        190: ".",
        188: ",",
        186: ":",
        222: "'",
        191: "\\",
        220: "/",
        219: "(",
        221: ")",
        189: "-",
        187: "=",

        # Control keys
        32: "SPACE",
        8: "BACKSPACE",
        9: "TAB",
        13: "ENTER",
    }

    def vk_to_token(self, vk_str: str) -> str:
        try:
            return self.VK_TO_TOKEN.get(int(vk_str), "UNK")
        except ValueError:
            return "UNK"

    def generate_features_file(self):
        hold_features, flight_features, ud_features = self.read_human_hold_flight_features()
        return hold_features, flight_features, ud_features

    # def create_key_sequence(self, password: str):
    #     key_sequence = []
    #
    #     for symbol in password:
    #         key_sequence.append(symbol)
    #
    #     return key_sequence

    def read_human_hold_flight_features(self):
        hold_features = []
        dd_features = []
        ud_features = []
        count = 0
        count_a = 0

        for root, _, files in os.walk(self.dataset_dir):
            for file in files:
                if not file.endswith("HUMAN.csv"):
                    continue

                file_path = os.path.join(root, file)
                # print("READING FILE:" + file_path)

                rows = []
                hold_times = []
                dd_times = []
                ud_times = []

                with open(file_path, newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if len(row) < 3:
                            continue

                        key_vk = row[0].strip()
                        hold = float(row[1])
                        dd = float(row[2])
                        ud = hold - dd

                        rows.append((key_vk, hold, dd, ud))
                        hold_times.append(hold)
                        dd_times.append(dd)
                        ud_times.append(ud)

                if not hold_times or not dd_times or not ud_times:
                    print("NO FEATURES FOUND. EXITING...")
                    continue

                hold_min = min(hold_times)
                hold_max = max(hold_times)
                denom_hold = hold_max - hold_min if hold_max != hold_min else 1.0

                dd_min = min(dd_times)
                dd_max = max(dd_times)
                denom_dd = dd_max - dd_min if dd_max != dd_min else 1.0

                ud_min = min(ud_times)
                ud_max = max(ud_times)
                denom_ud = ud_max - ud_min if ud_max != ud_min else 1.0

                for key_vk, hold, dd, ud in rows:
                    key = self.vk_to_token(key_vk).lower()
                    # print("KEY READ: " + key)
                    if key == ".":
                        count = count + 1
                    if key == "a":
                        count_a = count_a + 1
                    hold_norm = (hold - hold_min) / denom_hold
                    dd_norm = (dd - dd_min) / denom_dd
                    ud_norm = (ud - ud_min) / denom_ud
                    hold_features.append((key, hold_norm))
                    dd_features.append((key, dd_norm))
                    ud_features.append((key, ud_norm))

        # print("READ HOLD FEATURES FROM FILE: ")
        # print(hold_features)
        #
        # print("READ DD FEATURES FROM FILE: ")
        # print(dd_features)
        #
        # print("READ UD FEATURES FROM FILE: ")
        # print(ud_features)
        #
        # print("COUNT OF DOTS: ")
        # print(count)
        #
        # print("COUNT OF A: ")
        # print(count_a)
        return hold_features, dd_features, ud_features

# generator = SyntheticFeaturesGenerator("test_generate_other_users")
# dataset_reader = KeystrokeDatasetReader("test_generate_other_users")
# hold_features, flight_features, ud_features = dataset_reader.generate_features_file()
# generator.generate_other_users(hold_features, flight_features, ud_features)