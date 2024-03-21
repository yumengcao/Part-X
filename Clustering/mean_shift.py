import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
def mean_shift_(score: dict) -> np.ndarray:
        
    data = np.array(list(score.values())).reshape(-1,1)
    print(data)
    # 通过下列代码可自动检测bandwidth值
    # 从data中随机选取1000个样本，计算每一对样本的距离，然后选取这些距离的0.2分位数作为返回值，当n_samples很大时，这个函数的计算量是很大的。
    bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples= round(len(data)/2))
    print(bandwidth)
    # bin_seeding设置为True就不会把所有的点初始化为核心位置，从而加速算法
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    # 计算类别个数
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters)
    return labels