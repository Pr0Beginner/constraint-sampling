import numpy as np

class FactorEncoder:
    """因子编码转换器（修复离散变量处理逻辑）"""

    def __init__(self, factors):
        if not all(isinstance(v, dict) for v in factors.values()):
            raise ValueError("输入不符合规范")

        self.factors = factors
        self.dim_types = []
        self.cont_pos = {}  # 连续变量位置
        self.disc_pos = {}  # 离散变量位置（包含整数和分类）
        self.factor_names = list(self.factors.keys())  # 保持因子顺序
        self.factor_info_by_index = {}  # 用于按索引快速查找信息
        self.total_dims = 0
        self.n_cont = 0
        self.n_disc = 0
        self._parse_factors()

    def _parse_factors(self):
        """修复维度计算错误，正确统计离散变量数量"""
        pos = 0
        # 连续变量处理
        for name, info in self.factors.items():
            if info['type'] == 'cont':
                self.cont_pos[name] = pos
                self.factor_info_by_index[pos] = {'name': name, 'info': info}
                pos += 1
                self.n_cont += 1

        # 离散变量处理（修复n_disc计数错误）
        self.n_disc = 0  # 重置计数器
        for name, info in self.factors.items():
            if info['type'] == 'int':
                self.disc_pos[name] = {
                    'index': pos,
                    'type': 'int',
                    'bounds': info['bounds']
                }
                self.factor_info_by_index[pos] = {'name': name, 'info': info}
                pos += 1
                self.n_disc += 1  # 仅在识别到离散变量时计数
            elif info['type'] == 'cat':
                self.disc_pos[name] = {
                    'index': pos,
                    'type': 'cat',
                    'options': info['options']
                }
                self.factor_info_by_index[pos] = {'name': name, 'info': info}
                pos += 1
                self.n_disc += 1  # 仅在识别到离散变量时计数
        # 总维度
        self.total_dims = pos
        self.dim_types = [self.factor_info_by_index[i]['info']['type'] for i in range(self.total_dims)]

    def encode(self, sample):
        encoded = np.zeros(self.total_dims)
        # 连续变量部分（标准化到[0,1]）
        for name, idx in self.cont_pos.items():
            low, high = self.factors[name]['bounds']
            encoded[idx] = (sample[name] - low) / (high - low) if high > low else 0

        # 离散变量部分（整数直接存储，分类转索引）
        for name, info in self.disc_pos.items():
            if info['type'] == 'cat':
                encoded[info['index']] = info['options'].index(sample[name])
            else:  # int类型
                encoded[info['index']] = sample[name]
        return encoded

    def decode(self, encoded):
        decoded = {}
        # 连续变量解码（还原原始范围）
        for name, idx in self.cont_pos.items():
            low, high = self.factors[name]['bounds']
            decoded[name] = encoded[idx] * (high - low) + low

        # 离散变量解码（整数取整，分类转原值）
        for name, info in self.disc_pos.items():
            if info['type'] == 'cat':
                option_idx = np.clip(int(round(encoded[info['index']])), 0, len(info['options']) - 1)
                decoded[name] = info['options'][option_idx]
            else:  # int类型
                decoded[name] = np.clip(int(round(encoded[info['index']])), info['bounds'][0], info['bounds'][1])
        return decoded

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
                if self.factors[name]['type'] == 'cat':
                    decoded[i, j] = int(round(sample[self.disc_pos[name]['index']]))
                else:
                    decoded[i, j] = decoded_dict[name]
        return decoded

    def clip_and_round(self, sample):
        """确保离散变量严格在范围内，修复空数组处理"""
        if sample.size == 0:
            return sample  # 避免空数组操作

        for i in range(self.total_dims):
            var_info = self.factor_info_by_index[i]['info']
            if var_info['type'] == 'cont':
                sample[i] = np.clip(sample[i], 0, 1)
            elif var_info['type'] == 'int':
                low, high = var_info['bounds']
                sample[i] = np.round(sample[i])
                sample[i] = np.clip(sample[i], low, high)
            elif var_info['type'] == 'cat':
                num_options = len(var_info['options'])
                sample[i] = np.round(sample[i])
                sample[i] = np.clip(sample[i], 0, num_options - 1)
        return sample
