import statistics as stat
import random
import numpy as np

class SyntheticFeaturesGenerator:
    def __init__(self, username=None):
        self.username = username
        self.genuine_features = []
        self.generated_features = []
        self.context_sets = {}
        self.context_order = 3
        self.min_context_cardinality = 10
        self.channels = {"hold" : "hold", "UD" : "UD", "DD" : "DD"}

        self.init_context_sets()

    def set_username(self, username: str):
        self.username = username

    def init_context_sets(self):
        self.context_sets = {
            "hold": {},
            "DD": {},
            "UD": {}
        }

    def split_k1_k2_from_body(self, body: str):
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
                # DD.k1.k2 => key = k2
                _, key = self.split_k1_k2_from_body(feature_name[3:])
                dd_flight_features.append((key, value))

            elif feature_name.startswith("UD."):
                # UD.k1.k2 => key = k2
                _, key = self.split_k1_k2_from_body(feature_name[3:])
                ud_flight_features.append((key, value))

        return hold_features, dd_flight_features, ud_flight_features

    # this function may provide optimization to Si sets extraction, should be used later
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


    def fcm_sequence_si(self, training_u, m, context_tuple, target_key):
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

        n = len(training_u)
        for idx in range(n):
            if idx < m:
                continue

            key_j, val_j = training_u[idx]
            if key_j != target_key:
                continue

            if m == 0:
                prev_context = tuple()
            else:
                prev_context = tuple(training_u[idx - m + k][0] for k in range(m))

            if prev_context == context_tuple:
                S.add(val_j)

        return S

    def select_contexts_for_sequence(self, training_U, K_sequence, M, channel='hold'):
        """
        K_sequence : list of key names (target sequence) e.g. ['p','a','s','s','w','o','r','d']
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
        n = len(K_sequence)
        decisions = []

        for i in range(n):
            if (channel == 'UD' or channel == 'DD') and i == 0:
                decisions.append({
                    "index": i,
                    "key": K_sequence[i],
                    "chosen_order": -1,
                    "Si": set(),
                    "channel": channel,
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

                Si = self.fcm_sequence_si(training_U, m, context_tuple, target_key)

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
                "channel": channel,
                "use_random_fallback": use_random
            })

        return decisions


    def f_generating_func_mean(self, Si):
        """
        Implements the simplest method described in the paper:
        f(S_i) = average over all past observations in S_i.

        Si : set of timing values (non-empty)
        returns: float (mean timing)
        """
        return stat.mean(Si)

    def f_generating_func_icdf(self, Si: set[float]) -> float:
        """
        Sample a timing value from the inverse CDF of the empirical distribution in Si.

        Si : set of timing values (non-empty)
        returns: float (sampled timing)
        """
        if not Si:
            raise ValueError("Si cannot be empty")

        # 1. Sort the values
        sorted_values = np.sort(np.array(list(Si)))

        # 2. Build empirical CDF points (normalized)
        n = len(sorted_values)
        cdf = np.linspace(0, 1, n, endpoint=False) + 1 / (2 * n)  # center points in [0,1)

        # 3. Sample a uniform random number in [0,1)
        u = np.random.uniform(0, 1)

        # 4. Interpolate inverse CDF
        sampled_value = np.interp(u, cdf, sorted_values)
        return sampled_value

    def generate_features_from_decisions(self, decisions: list[dict], key_sequence: list[str], channel: str, generating_function: str = "icdf", repetitions: int = 1) -> list[tuple[str, float]]:
        generated_value = 0
        for d in decisions:
            feature_name = self.construct_feature_name(key_sequence=key_sequence, channel=channel, key_idx=d["index"])
            for rep in range(repetitions):
                if d["use_random_fallback"]:
                    generated_value = (feature_name, random.uniform(0, 0.3))
                elif generating_function == "mean":
                    generated_value = (feature_name, self.f_generating_func_mean(d["Si"]))
                elif generating_function == "icdf":
                    generated_value = (feature_name, self.f_generating_func_icdf(d["Si"]))
                self.generated_features[rep].append(generated_value)

    def construct_feature_name(self, key_sequence: list[str], channel: str, key_idx: int) -> str:
        if channel == 'hold':
            return "H." + key_sequence[key_idx]
        elif channel == 'UD':
            prev_key = key_sequence[key_idx - 1]
            curr_key = key_sequence[key_idx]
            return f"UD.{prev_key}.{curr_key}"
        elif channel == 'DD':
            prev_key = key_sequence[key_idx - 1]
            curr_key = key_sequence[key_idx]
            return f"DD.{prev_key}.{curr_key}"
        return ""


    def generate(self, hold_features: list[tuple[str, float]] = None, dd_flight_features: list[tuple[str, float]] = None, ud_flight_features: list[tuple[str, float]] = None, repetitions: int = 1) -> list[list[tuple[str, float]]]:
        # self.build_context_sets()

        if hold_features is None or dd_flight_features is None or ud_flight_features is None:
            hold_features, dd_flight_features, ud_flight_features = self.split_genuine_features()

        # print("FEATURE SPLIT SUCCESSFULLY")

        self.generated_features = [[] for _ in range(repetitions)]

        key_sequence = [".","t", "i", "e", "5", "Shift.r", "o", "a", "n", "l"]

        decisions_hold = self.select_contexts_for_sequence(hold_features, K_sequence=key_sequence,
                                                            M=self.context_order, channel=self.channels["hold"])

        # print("HOLD CONTEXT SETS CREATED SUCCESSFULLY")
        decisions_dd = self.select_contexts_for_sequence(dd_flight_features, K_sequence=key_sequence,
                                                            M=self.context_order, channel=self.channels["DD"])
        # print("DD CONTEXT SETS CREATED SUCCESSFULLY")

        decisions_ud = self.select_contexts_for_sequence(ud_flight_features, K_sequence=key_sequence,
                                                            M=self.context_order, channel=self.channels["UD"])

        # print("UD CONTEXT SETS CREATED SUCCESSFULLY")

        self.generate_features_from_decisions(decisions_hold, key_sequence=key_sequence,
                                              channel=self.channels["hold"],generating_function="icdf",
                                              repetitions=repetitions)

        # print("HOLD FEATURES GENERATED SUCCESSFULLY")
        self.generate_features_from_decisions(decisions_dd, key_sequence=key_sequence,
                                              channel=self.channels["DD"], generating_function="icdf",
                                              repetitions=repetitions)

        # print("DD FEATURES GENERATED SUCCESSFULLY")
        self.generate_features_from_decisions(decisions_ud, key_sequence=key_sequence,
                                              channel=self.channels["UD"], generating_function="icdf",
                                              repetitions=repetitions)

        # print("UD FEATURES GENERATED SUCCESSFULLY")

        # print("DECISIONS FOR HOLD KEY SEQUENCE: ")
        # print(decisions_hold)

        self.order_generated_features(key_sequence=key_sequence, repetitions=repetitions)

        print("Generated features in feature generator final:")
        print(self.generated_features)

        return self.generated_features

    def order_generated_features(self, key_sequence: list[str], repetitions: int = 1):
        """
        Reorders features so that for each key in K_sequence:
          H.key, DD.prev.curr, UD.prev.curr
        are grouped together.
        """
        for rep in range(repetitions):
            hold_dict = {name: val for name, val in self.generated_features[rep] if name.startswith("H.")}
            dd_dict = {name: val for name, val in self.generated_features[rep]  if name.startswith("DD.")}
            ud_dict = {name: val for name, val in self.generated_features[rep]  if name.startswith("UD.")}

            ordered_features = []

            for idx, curr_key in enumerate(key_sequence):
                if idx == 0:
                    hold_name = f"H.{curr_key}"
                    if hold_name in hold_dict:
                        ordered_features.append((hold_name, hold_dict[hold_name]))

                if idx > 0:
                    prev_key = key_sequence[idx - 1]
                    dd_name = f"DD.{prev_key}.{curr_key}"
                    if dd_name in dd_dict:
                        ordered_features.append((dd_name, dd_dict[dd_name]))

                    prev_key = key_sequence[idx - 1]
                    ud_name = f"UD.{prev_key}.{curr_key}"
                    if ud_name in ud_dict:
                        ordered_features.append((ud_name, ud_dict[ud_name]))

                    hold_name = f"H.{curr_key}"
                    if hold_name in hold_dict:
                        ordered_features.append((hold_name, hold_dict[hold_name]))

            self.generated_features[rep] = ordered_features
            # print(f"ORDERED FEATURES FOR ROW {rep}:")
            # print(ordered_features)



    def clear_data(self):
        self.genuine_features = []
        self.generated_features = []
        self.init_context_sets()


    # def generate_other_users(self, hold_features, dd_flight_features, ud_flight_features):
    #     # self.build_context_sets()
    #     #hold_features, dd_flight_features, ud_flight_features = self.split_genuine_features()
    #     # K_sequence = [".","a", "h", "s", "o", "Shift.d", "e", "c", "a", "s"]
    #     K_sequence = [".", "a", "h", "s", "o", "d", "e", "c", "a", "s"]
    #     channel_hold = 'hold'
    #     decisions_hold = self.select_contexts_for_sequence(hold_features, K_sequence=K_sequence, M=self.context_order,channel=channel_hold)
    #     self.generated_features.extend(self.generate_features_from_decisions(decisions_hold, K_sequence=K_sequence, channel=channel_hold,generating_function="icdf"))
    #     print("DECISIONS FOR HOLD KEY SEQUENCE: ")
    #     print(decisions_hold)
    #     # print("COUNT OF DECISIONS FOR DOT: ")
    #     # dot_decision_count = next(len(d["Si"]) for d in decisions_hold if d["key"] == ".")
    #     # print(dot_decision_count)
    #     # print("COUNT OF DECISIONS FOR A: ")
    #     # a_decision_count = next(len(d["Si"]) for d in decisions_hold if d["key"] == "a")
    #     # print(a_decision_count)
    #     channel_dd = 'DD'
    #     decisions_dd = self.select_contexts_for_sequence(dd_flight_features, K_sequence=K_sequence,M=self.context_order, channel=channel_dd)
    #     self.generated_features.extend(self.generate_features_from_decisions(decisions_dd, K_sequence=K_sequence, channel=channel_dd, generating_function="icdf"))
    #
    #     channel_ud = 'UD'
    #     decisions_ud = self.select_contexts_for_sequence(ud_flight_features, K_sequence=K_sequence, M=self.context_order, channel=channel_ud)
    #     self.generated_features.extend(self.generate_features_from_decisions(decisions_ud, K_sequence=K_sequence, channel=channel_ud, generating_function="icdf"))
    #
    #     ordered_features = self.order_generated_features(K_sequence=K_sequence)
    #
    #     print("Generated features in feature generator final:")
    #     print(self.generated_features)
    #
    #     return dict(ordered_features)



