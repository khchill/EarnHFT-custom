import sys
sys.path.append(".")
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy.stats import iqr
from sklearn.model_selection import GridSearchCV

def get_silverman_bandwidth(data):
    """ tính Silverman để ước lượng mật độ"""
    std_dev = np.std(data)
    interquartile_range = iqr(data)
    n = len(data)
    bandwidth = 1.06 * min(std_dev, interquartile_range / 1.34) * n ** (-1 / 5)
    return max(bandwidth, 1e-5)

def get_transformation_even_based_boltzmann_risk(
    buy_hold_return_list, bandwidth=None, kernel="gaussian", beta=10, risk_bond=0.1
):
    """ phân bổ lại ưu tiên lấy mẫu dữ liệu huấn luyện"""
    lower_risk_bond = risk_bond / 2
    upper_risk_bond = 1 - risk_bond / 2
    upper_value = np.quantile(buy_hold_return_list, upper_risk_bond)
    lower_value = np.quantile(buy_hold_return_list, lower_risk_bond)
    
    if bandwidth is None:
        silverman_bandwidth = get_silverman_bandwidth(buy_hold_return_list)
        log_bandwidths = np.linspace(np.log10(0.01 * silverman_bandwidth), np.log10(10 * silverman_bandwidth), 100)
        bandwidths = 10**log_bandwidths
        kde = KernelDensity(kernel=kernel)
        n_splits = min(5, len(buy_hold_return_list))
        if n_splits < 2:
            bandwidth = silverman_bandwidth
        else:
            grid = GridSearchCV(kde, {"bandwidth": bandwidths}, cv=n_splits)
            grid.fit(np.array(buy_hold_return_list).reshape(-1, 1))
            bandwidth = grid.best_params_["bandwidth"]
        
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(np.array(buy_hold_return_list)[:, np.newaxis])
    log_density = kde.score_samples(np.array(buy_hold_return_list)[:, np.newaxis])
    density = np.exp(log_density)
    density = density / np.sum(density)
    
    weights = []
    for return_rate, single_density in zip(buy_hold_return_list, density):
        if return_rate >= lower_value and return_rate <= upper_value:
            weights.append(1 / single_density)
        else:
            weights.append(1)
    
    weights = np.array(weights)
    final_weights = []
    # áp dụng trọng số ưu tiên Boltzmann với hệ số Beta
    for return_rate, weight in zip(buy_hold_return_list, weights):
        final_weights.append(weight * np.exp(beta * return_rate))
    return final_weights

from RL.util.sum_tree import SumTree

class start_selector(object):
    def __init__(self, start_list, initial_priority_list):
        self.start_list = start_list
        self.start_index_list = range(len(start_list))
        self.current_size = len(start_list)
        self.tree = SumTree(len(self.start_list))
        self.initial_priority_list = initial_priority_list
        
        for i in self.start_index_list:
            self.tree.update(i, initial_priority_list[i])

    def sample(self):
        v = np.random.uniform(0, self.tree.total_p)
        tree_idx, priority, data_idx = self.tree.get_leaf(v)
        return self.start_list[data_idx], priority, tree_idx

    def update(self, tree_idx, priority):
        self.tree.update(tree_idx, priority)
# 
def get_transformation_even_risk(buy_hold_return_list, bandwidth=None, kernel="gaussian", beta=None, risk_bond=0.1):
    """
    Hàm này tương tự như get_transformation_even_based_boltzmann_risk ở trên,
    nhưng KHÔNG nhân thêm phân phối Boltzmann (Beta).
    Chỉ dùng đơn thuần trọng số nghịch đảo KDE (Đoạn hiếm thì được chọn nhiều hơn).
    """
    lower_risk_bond = risk_bond / 2
    upper_risk_bond = 1 - risk_bond / 2
    upper_value = np.quantile(buy_hold_return_list, upper_risk_bond)
    lower_value = np.quantile(buy_hold_return_list, lower_risk_bond)
    
    if bandwidth is None:
        silverman_bandwidth = get_silverman_bandwidth(buy_hold_return_list)
        log_bandwidths = np.linspace(np.log10(0.01 * silverman_bandwidth), np.log10(10 * silverman_bandwidth), 100)
        bandwidths = 10**log_bandwidths
        kde = KernelDensity(kernel=kernel)
        n_splits = min(5, len(buy_hold_return_list))
        if n_splits < 2:
            bandwidth = silverman_bandwidth
        else:
            grid = GridSearchCV(kde, {"bandwidth": bandwidths}, cv=n_splits)
            grid.fit(np.array(buy_hold_return_list).reshape(-1, 1))
            bandwidth = grid.best_params_["bandwidth"]
        
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(np.array(buy_hold_return_list)[:, np.newaxis])
    log_density = kde.score_samples(np.array(buy_hold_return_list)[:, np.newaxis])
    density = np.exp(log_density)
    density = density / np.sum(density)
    
    weights = []
    for return_rate, single_density in zip(buy_hold_return_list, density):
        if return_rate >= lower_value and return_rate <= upper_value:
            weights.append(1 / single_density)
        else:
            weights.append(1)
            
    return weights

def get_transformation_even_based_sigmoid_risk(buy_hold_return_list, *args, **kwargs):
    return np.ones(len(buy_hold_return_list))
