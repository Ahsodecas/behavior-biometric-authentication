import statistics as stat
import random

class SyntheticFeaturesGenerator:
    def __init__(self, username=None):
        self.username = username
        self.genuine_features = []
        self.generated_features = []
        self.context_sets = {}
        self.init_context_sets()
        self.context_order = 3
        self.min_context_cardinality = 10


    def init_context_sets(self):
        self.context_sets = {
            "hold": {},
            "DD": {},
            "UD": {}
        }

    def _split_k1_k2_from_body(self, body: str):
        if not isinstance(body, str) or body == "":
            return None, None

        if body.endswith("."):
            k2 = "."
            rest = body[:-1]
            if rest.endswith("."):
                rest = rest[:-1]
            k1 = rest
            return k1, k2

        if "." in body:
            k1, k2 = body.rsplit(".", 1)
            return k1, k2

        return None, None

    def split_genuine_features(self):
        hold_features = []
        dd_flight_features = []
        ud_flight_features = []

        for feature_name, value in self.genuine_features:

            if feature_name.startswith("H."):
                # H.x => key = x
                key = feature_name[2:]
                hold_features.append((key, value))

            elif feature_name.startswith("DD."):
                # parts = feature_name.split(".")
                # DD.k1.k2 => key = k2
                # key = parts[-1]
                _, key = self._split_k1_k2_from_body(feature_name[3:])
                dd_flight_features.append((key, value))

            elif feature_name.startswith("UD."):
                # parts = feature_name.split(".")
                # UD.k1.k2 => key = k2
                # key = parts[-1]
                _, key = self._split_k1_k2_from_body(feature_name[3:])
                ud_flight_features.append((key, value))

        return hold_features, dd_flight_features, ud_flight_features

    def _build_context_set(self, features_dict, context_dict):
        for order in range(self.context_order + 1):
            for i in range(len(features_dict)):
                key_i, _ = features_dict[i]
                if i < order:
                    continue
                keys_context = tuple(features_dict[i - order + k][0] for k in range(order))
                context_set_i = set()
                for j in range(order, len(features_dict)):
                    prev_context = tuple(features_dict[j - order + k][0] for k in range(order))
                    if prev_context == keys_context:
                        context_set_i.add(features_dict[j][1])
                context_dict[order].setdefault(key_i, set()).update(context_set_i)


    def build_context_sets(self):
        hold_features, dd_flight_features, ud_flight_features = self.split_genuine_features()
        self._build_context_set(hold_features, self.context_sets["hold"])
        self._build_context_set(dd_flight_features,  self.context_sets["DD"])
        self._build_context_set(ud_flight_features, self.context_sets["UD"])


    def FCM_sequence_Si(self, training_U, m, context_tuple, target_key):
        """
        training_U : [(key, t), ...]  chronological sequence across all training samples
        m          : context order (integer >= 0)
        context_tuple : tuple of preceding keys of length m
        target_key : the key whose S_i we want (string)

        returns: set of timing values observed for target_key when preceding context == context_tuple
        """
        S = set()
        if m < 0:
            return S

        n = len(training_U)
        for idx in range(n):
            if idx < m:
                continue

            key_j, val_j = training_U[idx]
            if key_j != target_key:
                continue

            if m == 0:
                prev_context = tuple()
            else:
                prev_context = tuple(training_U[idx - m + k][0] for k in range(m))

            if prev_context == context_tuple:
                S.add(val_j)

        return S

    def select_contexts_for_sequence(self, training_U, K_sequence, M, channel='hold'):
        """
        K_sequence : list of key names (target sequence) e.g. ['h','e','l','l','o']
        In this context - the symbols of the password
        M          : maximum context order (int)
        channel    : 'hold' or 'UD' or 'DD'  (what we are synthesizing here)
        training_sequences :
            dict with keys:
              'hold' -> [(key, t), ...]
              'UD'   -> [(key, t), ...]  (key is the destination key, consistent with FCM)
              'DD'   -> [(key, t), ...]
            (these sequences must be chronological concatenation of all training samples)

        Returns:
            decisions : list of length len(K_sequence)
                Each element is a dict:
                {
                  "index": i,
                  "key": K_sequence[i],
                  "chosen_order": m_used,   # -1 means fallback to random
                  "Si": set(...) or set(),
                  "use_random_fallback": bool
                }
        """
        # if channel == 'hold':
        #     training_U = self.context_sets.get('hold', [])
        # elif channel == 'UD':
        #     training_U = self.context_sets.get('UD', [])
        # elif channel == 'DD':
        #     training_U = self.context_sets.get('DD', [])
        # else:
        #     raise ValueError("channel must be 'hold', 'DD' or 'UD'")

        n = len(K_sequence)
        decisions = []

        for i in range(n):
            # for flights: there is no flight before the first key (i == 0)
            if (channel == 'UD' or channel == 'DD') and i == 0:
                decisions.append({
                    "index": i,
                    "key": K_sequence[i],
                    "chosen_order": -1,
                    "Si": set(),
                    "use_random_fallback": True
                })
                continue

            m = min(M, i)

            chosen_Si = set()
            chosen_m = -1
            use_random = False
            target_key = K_sequence[i]

            while m >= 0:
                if m == 0:
                    context_tuple = tuple()
                else:
                    # slice: K[i-m : i] are the m preceding keys
                    context_tuple = tuple(K_sequence[i - m: i])

                Si = self.FCM_sequence_Si(training_U, m, context_tuple, target_key)

                if len(Si) < self.min_context_cardinality:
                    m = m - 1
                    continue
                else:
                    chosen_Si = Si
                    chosen_m = m
                    break

            if chosen_m >= 0:
                use_random = False
            else:
                chosen_Si = set()
                chosen_m = -1
                use_random = True

            decisions.append({
                "index": i,
                "key": target_key,
                "chosen_order": chosen_m,
                "Si": chosen_Si,
                "use_random_fallback": use_random
            })

        return decisions


    def f_mean(self, Si):
        """
        Implements the simplest method described in the paper:
        f(S_i) = average over all past observations in S_i.

        Si : set of timing values (non-empty)
        returns: float (mean timing)
        """
        return stat.mean(Si)

    def generate_features_from_decisions(self, decisions):
        generate_values = []
        for d in decisions:
            if d["use_random_fallback"]:
                generate_values.append((d["key"], random.uniform(0, 1500)))
            else:
                generate_values.append((d["key"], stat.mean(d["Si"])))
        return generate_values


    def generate(self):
        # self.build_context_sets()
        hold_features, dd_flight_features, ud_flight_features = self.split_genuine_features()
        K_sequence = [".","t", "i", "e", "5", "R", "o", "a", "n", "l"]
        decisions = self.select_contexts_for_sequence(hold_features, K_sequence=K_sequence, M=self.context_order, channel='hold')
        self.generated_features.extend(self.generate_features_from_decisions(decisions))
        print("Generated features:")
        print(self.generated_features)


    def clear_data(self):
        self.genuine_features = []
        self.generated_features = []
        self.init_context_sets()
        self.features_folder_path = ""



