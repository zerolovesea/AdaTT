import cityhash
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def sample_pos_neg(data, target_col, neg_ratio=0.5, random_state=42):
    pos = data[data[target_col] == 1]
    neg = data[data[target_col] == 0]
    n_pos = len(pos)
    n_neg = min(len(neg), int(round(n_pos * neg_ratio)))
    neg_sampled = neg.sample(n=n_neg, random_state=random_state)
    return pd.concat([pos, neg_sampled])

def ks_score(y_true, y_pred):
    """计算KS值"""
    return ks_2samp(y_pred[y_true == 1], y_pred[y_true == 0]).statistic

def hash_mapping(value, vocab_size):
    """预处理字符串并hash到指定vocab_size范围，缺失值返回0"""
    try:
        if pd.isna(value):
            processed = '<MISSING>'
        else:
            processed = str(value).strip().lower().replace('-', '_')
        return cityhash.CityHash32(processed.encode()) % vocab_size
    except Exception:
        return 0
    
def parse_varlen(x):
    """解析变长特征，返回list[str/int]"""
    if pd.isna(x):
        return []
    if isinstance(x, (list, np.ndarray)):
        return list(x)
    if isinstance(x, str):
        return [i.strip() for i in x.split(',') if i.strip()]
    return [str(x)]

def compute_sample_weights(data, target_cols):
    weights = np.ones((len(data), len(target_cols)), dtype=np.float32)
    for idx, col in enumerate(target_cols):
        vc = data[col].value_counts()
        n_pos = vc.get(1, 0)
        n_neg = vc.get(0, 0)
        n_total = n_pos + n_neg
        w_pos = n_total / (2 * n_pos) if n_pos > 0 else 1.0
        w_neg = n_total / (2 * n_neg) if n_neg > 0 else 1.0
        weights[:, idx] = data[col].map({1: w_pos, 0: w_neg}).values
    return weights