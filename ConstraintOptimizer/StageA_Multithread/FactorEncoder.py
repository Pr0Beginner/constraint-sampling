import numpy as np


class FactorEncoder:
    def __init__(self, factors, decimal_places=4):
        if not all(isinstance(v, dict) for v in factors.values()):
            raise ValueError("Invalid factor configuration")

        self.factors = factors
        self.decimal_places = decimal_places
        self.factor_names = list(self.factors.keys())

        self.dim_types = []
        self.cont_pos = {}
        self.disc_pos = {}
        self.factor_info_by_index = {}

        self.total_dims = 0
        self.n_cont = 0
        self.n_disc = 0

        self._cont_meta = []
        self._int_meta = []
        self._cat_meta = []
        self._parse_factors()

    def _parse_factors(self):
        pos = 0
        for name, info in self.factors.items():
            if info["type"] == "cont":
                self.cont_pos[name] = pos
                self.factor_info_by_index[pos] = {"name": name, "info": info}
                low, high = info["bounds"]
                self._cont_meta.append((name, pos, low, high, high - low))
                pos += 1
                self.n_cont += 1

        self.n_disc = 0
        for name, info in self.factors.items():
            if info["type"] == "int":
                self.disc_pos[name] = {"index": pos, "type": "int", "bounds": info["bounds"]}
                self.factor_info_by_index[pos] = {"name": name, "info": info}
                low, high = info["bounds"]
                self._int_meta.append((name, pos, low, high))
                pos += 1
                self.n_disc += 1
            elif info["type"] == "cat":
                self.disc_pos[name] = {"index": pos, "type": "cat", "options": info["options"]}
                self.factor_info_by_index[pos] = {"name": name, "info": info}
                options = info["options"]
                self._cat_meta.append((name, pos, options, len(options) - 1))
                pos += 1
                self.n_disc += 1

        self.total_dims = pos
        self.dim_types = [self.factor_info_by_index[i]["info"]["type"] for i in range(self.total_dims)]

    def encode(self, sample):
        encoded = np.zeros(self.total_dims)
        for name, idx, low, _, span in self._cont_meta:
            encoded[idx] = (sample[name] - low) / span if span > 0 else 0.0
        for name, idx, _, _ in self._int_meta:
            encoded[idx] = sample[name]
        for name, idx, options, _ in self._cat_meta:
            encoded[idx] = options.index(sample[name])
        return encoded

    def decode(self, encoded):
        decoded = {}
        for name, idx, low, _, span in self._cont_meta:
            decoded[name] = encoded[idx] * span + low
        for name, idx, low, high in self._int_meta:
            decoded[name] = np.clip(int(round(encoded[idx])), low, high)
        for name, idx, options, max_idx in self._cat_meta:
            option_idx = np.clip(int(round(encoded[idx])), 0, max_idx)
            decoded[name] = options[option_idx]
        return decoded

    def clip_and_round(self, sample):
        if sample.size == 0:
            return sample

        for i in range(self.total_dims):
            var_info = self.factor_info_by_index[i]["info"]
            var_type = var_info["type"]
            if var_type == "cont":
                sample[i] = np.clip(sample[i], 0, 1)
            elif var_type == "int":
                low, high = var_info["bounds"]
                sample[i] = np.clip(sample[i], low, high)
            elif var_type == "cat":
                sample[i] = np.clip(sample[i], 0, len(var_info["options"]) - 1)
        return sample

    def decode_single_sample(self, sample):
        decoded_dict = self.decode(sample)
        return [decoded_dict[name] for name in self.factor_names]

    def decode_to_arr(self, encoded_samples):
        if encoded_samples.ndim == 1:
            encoded_samples = encoded_samples.reshape(1, -1)
        decoded = np.empty((len(encoded_samples), len(self.factors)), dtype=object)
        for i, sample in enumerate(encoded_samples):
            decoded_dict = self.decode(sample)
            for j, name in enumerate(self.factor_names):
                decoded[i, j] = decoded_dict[name]
        return decoded

    def decode_to_number(self, encoded_samples):
        if encoded_samples.ndim == 1:
            encoded_samples = encoded_samples.reshape(1, -1)

        decoded = np.empty((len(encoded_samples), len(self.factors)))
        for i, sample in enumerate(encoded_samples):
            decoded_dict = self.decode(sample)
            for j, name in enumerate(self.factor_names):
                if self.factors[name]["type"] == "cat":
                    decoded[i, j] = int(round(sample[self.disc_pos[name]["index"]]))
                else:
                    decoded[i, j] = decoded_dict[name]
        return decoded
