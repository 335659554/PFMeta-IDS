import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import ConcatDataset
import pandas as pd
import os

n_clients = 20
dirichlet_alpha = 1.0
seed = 66
attack_type = 'rpm'


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    """
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    """
    n_classes = train_labels.nunique()
    # label_distribution: shape (K, N)，记录每个类别分到每个client的比例
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    # class_idcs: 长度为K的列表，每个元素是该类别的样本下标
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split 按 fracs 比例将 k_idcs 划分为n_clients个子集
        for i, idcs in enumerate(np.split(
                k_idcs,
                (np.cumsum(fracs)[:-1] * len(k_idcs)).astype(int))):
            client_idcs[i] += [idcs]
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs


if __name__ == "__main__":
    np.random.seed(seed)
    dataset = pd.read_csv(f"norm_{attack_type}.csv")

    labels = dataset['Label']

    # Non-IID划分
    client_idcs = dirichlet_split_noniid(labels, alpha=dirichlet_alpha, n_clients=n_clients)

    # 将每个client的数据存成一个CSV
    for client, idxs in enumerate(client_idcs):
        os.makedirs(f"clients_dataset/{attack_type}/alpha{dirichlet_alpha}/{client}", exist_ok=True)
        dataset.iloc[idxs].to_csv(
            f"clients_dataset/{attack_type}/alpha{dirichlet_alpha}/{client}/c{client}.csv",
            index=False
        )

    # 统计每个client不同label数量
    unique_labels = sorted(labels.unique())
    n_labels = len(unique_labels)
    counts = np.zeros((n_clients, n_labels))

    for i in range(n_clients):
        idxs = client_idcs[i]
        label_counts = labels[idxs].value_counts()
        for label_val, c in label_counts.items():
            col_index = unique_labels.index(label_val)
            counts[i, col_index] = c

    x = np.arange(n_clients)
    bottom = np.zeros(n_clients)

    plt.figure(figsize=(12, 8))

    # 根据标签值确定图例文本
    for j, label_val in enumerate(unique_labels):
        if int(label_val) == 0:
            bar_label = "normal message"
        else:
            bar_label = "attack message"
        plt.bar(
            x,
            counts[:, j],
            bottom=bottom,
            label=bar_label
        )
        bottom += counts[:, j]

    # 调整横纵轴刻度字体大小
    plt.xticks(x, [f"{i}" for i in range(n_clients)], fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlabel("Client", fontsize=14)
    plt.ylabel("Number of samples", fontsize=14)
    # plt.title(f"Message distribution on different clients(alpha={dirichlet_alpha})", fontweight="bold", fontsize=14)
    plt.title(f"$\\alpha={dirichlet_alpha}$", fontsize=14)


    # 先获取原图例句柄与文本，再将“attack message”放在前面（上方）
    handles, labels_text = plt.gca().get_legend_handles_labels()
    attack_idx = labels_text.index("attack message")
    normal_idx = labels_text.index("normal message")
    # 重新组装 legend
    new_handles = [handles[attack_idx], handles[normal_idx]]
    new_labels = [labels_text[attack_idx], labels_text[normal_idx]]

    plt.legend(new_handles, new_labels, loc="upper right", fontsize=13)

    plt.savefig(f"clients_dataset/{attack_type}/alpha{dirichlet_alpha}/dis{attack_type}a{dirichlet_alpha}.png", dpi=300)
    plt.show()
