import copy
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import nn, optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import math
import os
import pickle
from random import shuffle
import sys
import argparse
from imblearn.over_sampling import SMOTE  # 导入SMOTE
from imblearn.under_sampling import RandomUnderSampler


# ----------------------
# 全局超参数等配置
# ----------------------
attack_type = 'dos'
dirichlet_alpha = 0.5

# 分割出的任务集大小
split_size = 512
# 1,2,4,8
meta_batchsize = 1
# 1, 5, 10
N_U = 1

comm_round_n = 0

# ----------------------
# 客户端参数配置
# ----------------------
client = 0

# ----------------------
# 优化参数配置
# ----------------------
# test task ratio
ratio = 0.2
num_test_task = math.ceil(split_size * ratio)
feature_num = 9
class_num = 2
epoch_num = 1

# 优化器相关
beta = 0.01     # meta 学习率 (Adam)
weight_decay = 0.01

# 设定正则化参数 lambda
lambda_reg = 0.5  # 可以根据需要调整

# 这里将 alpha 固定为一个常量（不再学习）
fixed_alpha = 0.01  # 可以根据需要自行调参

csv_file = [
    'processed_Dos_data.csv',
    'processed_Fuzzy_data.csv',
    'processed_gear_data.csv',
    'processed_rpm_data.csv'
]

# ----------------------
# 工具函数
# ----------------------
def file_exists(filepath):
    return os.path.exists(filepath)

def sgd_optimize(paralist, lr, gradlist):
    """
    使用给定固定学习率 lr（标量），对当前模型参数进行一次更新:
        para.data = para.data - lr * grad
    """
    for para, grad in zip(paralist, gradlist):
        para.data -= lr * grad

def inisitagrad_add(a, b):
    """
    将两个列表里的梯度(张量)对应元素相加
    """
    return [x + y for x, y in zip(a, b)]

def apply_smote(dataset):
    """
    应用SMOTE到PyTorch的Dataset。
    """
    if len(dataset) == 0:
        print("[Warning] Empty dataset, cannot apply SMOTE.")
        return dataset

    # 提取所有数据
    X = torch.stack([item[0] for item in dataset]).numpy()
    y = torch.stack([item[1] for item in dataset]).numpy()

    # 应用SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # 转换回Tensor
    X_resampled = torch.tensor(X_resampled, dtype=torch.float32)
    y_resampled = torch.tensor(y_resampled, dtype=torch.long)

    return Traffic_Task_Dataset(x_data=X_resampled, y_data=y_resampled)


def apply_undersample(task_dataset):
    if len(task_dataset) == 0:
        print("[Warning] Empty dataset, cannot apply undersampling.")
        return task_dataset

    X = torch.stack([item[0] for item in task_dataset])
    y = torch.stack([item[1] for item in task_dataset])

    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X.numpy(), y.numpy())

    X_resampled = torch.tensor(X_resampled, dtype=torch.float32)
    y_resampled = torch.tensor(y_resampled, dtype=torch.long)

    # 返回一个与原始格式一致的 Traffic_Task_Dataset
    return Traffic_Task_Dataset(x_data=X_resampled, y_data=y_resampled)


# ----------------------
# 数据集类
# ----------------------
class Traffic_Task_Dataset(Dataset):
    """
    当加载以 .pt 文件存储的数据时，可使用本类。
    同时也支持直接传递 x_data, y_data 用于构造 Dataset。
    """

    def __init__(self,
                 attack=None,
                 no_task=None,
                 base_path=None,
                 x_data=None,
                 y_data=None):
        """
        两种使用方法：
        1. 直接从 .pt 文件中加载数据 (当 attack, no_task, base_path 有效时)。
        2. 直接传入 x_data 和 y_data (当 x_data, y_data 不为空时)。
        """

        # 当 x_data 和 y_data 同时提供时，优先使用内存数据
        if x_data is not None and y_data is not None:
            self.x_data = x_data
            self.y_data = y_data
            self.n_samples = len(self.x_data)
        else:
            # 否则尝试从 .pt 文件加载
            if attack is None or no_task is None or base_path is None:
                print("[Warning] attack/no_task/base_path 未指定，且 x_data, y_data 为空，无法加载数据。")
                self.x_data = []
                self.y_data = []
                self.n_samples = 0
            else:
                x_path = os.path.join(base_path, f'{attack}_x_data{no_task}.pt')
                y_path = os.path.join(base_path, f'{attack}_y_data{no_task}.pt')
                if os.path.exists(x_path) and os.path.exists(y_path):
                    self.x_data = torch.load(x_path)
                    self.y_data = torch.load(y_path)
                    self.n_samples = len(self.x_data)
                else:
                    print(f"[Warning] 未找到任务 {no_task} 的数据集文件: {attack}")
                    self.x_data = []
                    self.y_data = []
                    self.n_samples = 0

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# ----------------------
# 模型定义
# ----------------------
class Bottleneck(nn.Module):
    """
    Bottleneck结构:
      in_channels -> hidden_dim -> out_channels
      其中包含:
        1) 1x1 pointwise conv (扩张)
        2) 1x3 depthwise conv
        3) 1x1 pointwise conv (压缩)
      并在每层卷积后插入BN和ReLU6
    """
    def __init__(self, in_channels, hidden_dim, out_channels):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU6(inplace=True)

        # 深度卷积，groups 等于通道数
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                               padding=1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU6(inplace=True)

        self.conv3 = nn.Conv1d(hidden_dim, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.relu3 = nn.ReLU6(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        return out


class LWCNN(nn.Module):
    """
    根据题干描述的一维LW-CNN结构:
      1) 标准Conv1D(输入->32通道)
      2) DepthwiseConv1D(stride=2) + Bottleneck(64->32)
      3) Dropout
      4) DepthwiseConv1D(stride=2) + Bottleneck(128->32)
      5) Dropout
      6) Global Average Pooling
      7) 输出层：Sigmoid(二分类) 或 Softmax(多分类)
    """
    def __init__(self,
                 in_channels=1,      # 输入通道数，默认为单通道信号
                 num_classes=2,      # 输出类别数, 2表示二分类；>2表示多分类
                 dropout_p=0.2       # Dropout概率
                 ):
        super(LWCNN, self).__init__()
        self.num_classes = num_classes

        # 1) 标准卷积，将通道数扩展到32
        #   Conv1D 1×3×32
        self.conv_init = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        )

        # -----------------------
        # 第一次下采样 (DepthwiseConv, stride=2) + Bottleneck I (64->32)
        # -----------------------
        # depthwise conv: 32 -> 32, stride=2
        self.dwconv1 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, groups=32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU6(inplace=True)
        )
        # Bottleneck I: 32 -> 64 -> 32
        self.bottleneck1 = Bottleneck(in_channels=32, hidden_dim=64, out_channels=32)

        # 用于给跳跃连接的输入做维度匹配(同样stride=2)，以便加到Bottleneck输出上
        # self.skip_conv1 = nn.Conv1d(32, 32, kernel_size=1, stride=2, bias=False)

        self.dropout1 = nn.Dropout(p=dropout_p)

        # -----------------------
        # 第二次下采样 (DepthwiseConv, stride=2) + Bottleneck II (128->32)
        # -----------------------
        self.dwconv2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, groups=32, bias=False)
        )
        # Bottleneck II: 32 -> 128 -> 32
        self.bottleneck2 = Bottleneck(in_channels=32, hidden_dim=128, out_channels=32)

        # self.skip_conv2 = nn.Conv1d(32, 32, kernel_size=1, stride=2, bias=False)

        self.dropout2 = nn.Dropout(p=dropout_p)

        # Global Average Pooling: 将每个通道的序列特征压缩为单个数值
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)

        # 输出层：
        # 二分类 -> [N, 1] + Sigmoid
        # 多分类 -> [N, num_classes] + Softmax
        if num_classes == 2:
            self.classifier = nn.Sequential(
                nn.Conv1d(32, 1, kernel_size=1, stride=1, bias=True),
                nn.Sigmoid()  # 输出[batch_size, 1, 1]
            )
        else:
            self.classifier = nn.Sequential(
                nn.Conv1d(32, num_classes, kernel_size=1, stride=1, bias=True),
                nn.Softmax(dim=1)  # 输出[batch_size, num_classes, 1]
            )

    def forward(self, x):
        """
        x: [batch_size, in_channels, seq_len]
        """
        # (1) 标准卷积 => [N, 32, L]
        x = x.view(-1, 1, feature_num)
        x = self.conv_init(x)

        # (2) 第一次深度可分离卷积 => [N, 32, L/2]
        out1 = self.dwconv1(x)
        # Bottleneck I => [N, 32, L/2]
        shortcut1 = self.bottleneck1(out1)
        # 跳跃连接(残差)
        # shortcut1 = self.skip_conv1(x)  # [N, 32, L/2]
        out1 = out1 + shortcut1

        out1 = self.dropout1(out1)

        # (3) 第二次深度可分离卷积 => [N, 32, L/4]
        out2 = self.dwconv2(out1)
        # Bottleneck II => [N, 32, L/4]
        shortcut2 = self.bottleneck2(out2)
        # 跳跃连接(残差)
        # shortcut2 = self.skip_conv2(out1)  # [N, 32, L/4]
        out2 = out2 + shortcut2

        out2 = self.dropout2(out2)

        # (4) 全局平均池化 => [N, 32, 1]
        out2 = self.gap(out2)

        # (5) 输出层 => [N, 1] (二分类) 或 [N, num_classes] (多分类)
        out2 = self.classifier(out2)  # 形状：[N, 1, 1] 或 [N, num_classes, 1]
        out2 = out2.squeeze(-1)       # => [N, 1] 或 [N, num_classes]
        return out2




# ----------------------
# 主训练流程 (FO-MAML) —— 仅对初始参数进行学习，加入正则化项
# ----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_type', type=str, default='hybrid')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5)
    parser.add_argument('--meta_batchsize', type=int, default=1)
    parser.add_argument('--N_U', type=int, default=1)
    parser.add_argument('--comm_round_n', type=int, default=0)
    parser.add_argument('--client', type=int, default=0)
    parser.add_argument('--lambda_reg', type=float, default=0.5)
    parser.add_argument('--split_size', type=float, default=0.5)
    args = parser.parse_args()

    # 用解析到的参数值覆盖全局或局部变量
    attack_type = args.attack_type
    dirichlet_alpha = args.dirichlet_alpha
    meta_batchsize = args.meta_batchsize
    N_U = args.N_U
    comm_round_n = args.comm_round_n
    client = args.client
    split_size = int(args.split_size)
    num_test_task = int(split_size * ratio)
    lambda_reg = args.lambda_reg

    # dataset_path = f'clients_dataset/{attack_type}/alpha{dirichlet_alpha}/{client}/splitted_dataset/batch_size_{meta_batchsize}'
    dataset_path = f'clients_dataset/{attack_type}/alpha{dirichlet_alpha}/{client}/splitted_dataset'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)
    file = open(f'{dataset_path}/batch_size_{meta_batchsize}/LWso_l{lambda_reg}b{beta}.log', 'a')
    sys.stdout = file
    print(f"-----------round {comm_round_n}-----------")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    g_path = f'clients_dataset/ini_sita_g.pth'
    if comm_round_n == 0:
        pass
    else:
        g_path = f'clients_dataset/{attack_type}/alpha{dirichlet_alpha}/{client}/splitted_dataset/batch_size_{meta_batchsize}/LWso_sita_g.pth'

    # 加载任务数据集
    tasks_dataset = []
    for no_dataset in range(split_size):
        task_file_path = f'{dataset_path}/{attack_type}_data{no_dataset}.pkl'
        if file_exists(task_file_path):
            with open(task_file_path, 'rb') as f:
                task_dataset = pickle.load(f)
                tasks_dataset.append(task_dataset)
        else:
            print(f"[Warning] 未找到任务数据集文件: {task_file_path}")
            sys.exit(1)

    # 查看每个任务的数据量
    for i in range(len(tasks_dataset)):
        print(f"The length of task {i}'s dataset is {tasks_dataset[i].n_samples}")

    # 加载已有的 meta_model 或自行初始化
    sita_g = None
    sita_p = None
    if file_exists(g_path):
        sita_g = torch.load(g_path, map_location=device)
        print("Loaded existing g_model.")
    else:
        sita_g = LWCNN().to(device)
        torch.save(sita_g, g_path)
        print("Initialized a new g_model.")

    if file_exists(f'{dataset_path}/batch_size_{meta_batchsize}/LWso_sita_p.pth') and comm_round_n != 0:
        sita_p = torch.load(f'{dataset_path}/batch_size_{meta_batchsize}/LWso_sita_p.pth', map_location=device)
        print("Loaded existing p_model.")
    else:
        sita_p = copy.deepcopy(sita_g)
        print("Initialized a new p_model.")

    torch.save(sita_g, f'{dataset_path}/batch_size_{meta_batchsize}/LWso_sita_g.pth')
    torch.save(sita_p, f'{dataset_path}/batch_size_{meta_batchsize}/LWso_sita_p.pth')

    # 定义 meta 优化器 (仅对 meta_model.parameters() 优化)
    g_optimizer = optim.Adam(sita_g.parameters(), lr=beta, weight_decay=weight_decay)
    p_optimizer = optim.Adam(sita_p.parameters(), lr=beta, weight_decay=weight_decay)
    criterion = nn.BCELoss()


    # ----------------------
    # 训练 sita_g
    # ----------------------
    print("Start training sita_g...")
    for epoch in range(epoch_num):
        print(f"Epoch: {epoch}")

        # 用于累积对 meta_model 初始参数的梯度
        ini_sita_grad = None

        g_optimizer.zero_grad()

        # meta 批次统计
        cnt = 0
        step = 0

        # 1) 训练阶段：对训练任务（大约 80% 的任务）进行元训练
        train_tasks = tasks_dataset[: (len(tasks_dataset) - num_test_task)]
        for i, task_dataset in enumerate(train_tasks):
            task_dataset = apply_smote(task_dataset)
            try:
                train_dataset, test_dataset = train_test_split(
                    task_dataset, test_size=0.5, random_state=0, stratify=task_dataset.y_data
                )
            except Exception as e:
                # 如果 stratify 失败或者数据量不够，就跳过
                print(f"[Warning] Splitting task {i} failed: {e}, skip.")
                continue

            # 应用SMOTE到训练集
            # train_dataset = apply_smote(train_dataset)

            # 应用SMOTE到测试集（注意：通常不推荐对测试集应用SMOTE，但根据需求这里进行了）
            # test_dataset = test_dataset

            # 构建 DataLoader
            train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

            # 拷贝 meta_model，准备在该任务上做一次内循环训练
            model = copy.deepcopy(sita_g).to(device)
            model.train()

            # (1.1) 在训练集上做若干步训练 (此处只写单 epoch 遍历，你也可多次遍历 train_loader)
            train_loss_sum = 0.0
            for update_step in range(N_U):
                for inputs, labels in train_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device).long().view(-1)

                    # 清空梯度
                    for para in model.parameters():
                        para.grad = None

                    outputs = model(inputs)
                    loss = criterion(outputs, labels.float().view(-1, 1))

                    # 计算正则化项: 0.5 * lambda_reg * ||theta_p - theta_g||^2
                    # l2_reg = 0.5 * lambda_reg * sum(
                    #     (p_p - p_g).pow(2).sum() for p_p, p_g in zip(model.parameters(), sita_g.parameters())
                    # )

                    total_loss = loss
                    total_loss.backward()

                    # 用固定 alpha 来更新参数 (FO-MAML 内层)
                    grads = [p.grad for p in model.parameters()]
                    sgd_optimize(model.parameters(), fixed_alpha, grads)

                    # 仅累计分类损失，不包括正则化损失
                    train_loss_sum += loss.item()

            if len(train_loader) > 0:
                print(f"Train_task:{i}, Loss per batch:{train_loss_sum / len(train_loader):.4f}")
            else:
                print(f"Train_task:{i}, (train_loader is empty).")

            # (1.2) 使用更新后的 model 在测试集上计算 loss，并对 meta_model 的初始参数做一阶梯度
            model.eval()

            if len(test_loader) > 0:
                # 再 forward 一下整份测试集 loss
                test_loss_sum = 0.0
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device).long().view(-1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.float().view(-1, 1))
                    test_loss_sum += loss.item()

                print(f"Test_task:{i}, Loss per batch:{test_loss_sum / len(test_loader):.4f}")

                # ---- FO-MAML 核心：对“更新后参数”做一次一阶 backward ----
                # 重新取一次 test_loader 里的数据（这里简单地取第一个 batch）做 backward
                inputs_test, labels_test = next(iter(test_loader))
                inputs_test = inputs_test.to(device)
                labels_test = labels_test.to(device).long().view(-1)

                # 切回 train 模式以允许梯度
                model.train()
                for para in model.parameters():
                    para.grad = None

                outputs_test = model(inputs_test)
                loss_test = criterion(outputs_test, labels_test.float().view(-1, 1))

                # # 计算元更新的正则化项
                # l2_reg_meta = 0.5 * lambda_reg * sum(
                #     (p_p - p_g).pow(2).sum() for p_p, p_g in zip(model.parameters(), sita_g.parameters())
                # )

                # 总损失用于元更新
                total_loss_meta = loss_test
                total_loss_meta.backward()

                # model.parameters() 里 param.grad 就是一阶近似，用来更新 sita_g
                grads_now = [p.grad.clone() for p in model.parameters()]
                if ini_sita_grad is None:
                    ini_sita_grad = grads_now
                else:
                    ini_sita_grad = inisitagrad_add(ini_sita_grad, grads_now)

            if (i+1) % meta_batchsize == 0 or (i+1)==len(train_tasks):
                # (1.3) 根据 meta_batchsize 进行一次 meta update
                # 更新 meta_model 的初始参数
                if ini_sita_grad is not None:
                    for param, g in zip(sita_g.parameters(), ini_sita_grad):
                        param.grad = g
                    g_optimizer.step()
                # 清理
                g_optimizer.zero_grad()
                step += 1
                print(f"  -> Meta update step {step} done.")
                print('--------------------------------------------')

        # 可选：保存sita_g更新的梯度
        with open(f'{dataset_path}/batch_size_{meta_batchsize}/LWso_sita_g_grad.pkl', 'wb') as f:
            pickle.dump(ini_sita_grad, f)

    sita_g.eval()
    for param in sita_g.parameters():
        param.requires_grad = False

    # ----------------------
    # 训练 sita_p
    # ----------------------
    print("Start training sita_p...")
    for epoch in range(epoch_num):
        print(f"Epoch: {epoch}")

        # 用于累积对 sita_p 初始参数的梯度
        ini_sita_grad = None

        p_optimizer.zero_grad()

        # meta 批次统计
        step = 0

        # 1) 训练阶段：对训练任务（大约 80% 的任务）进行元训练
        train_tasks = tasks_dataset[: (len(tasks_dataset) - num_test_task)]
        for i, task_dataset in enumerate(train_tasks):
            task_dataset = apply_smote(task_dataset)
            try:
                train_dataset, test_dataset = train_test_split(
                    task_dataset, test_size=0.5, random_state=0, stratify=task_dataset.y_data
                )
            except Exception as e:
                # 如果 stratify 失败或者数据量不够，就跳过
                print(f"[Warning] Splitting task {i} failed: {e}, skip.")
                continue

            # 应用SMOTE到训练集
            # train_dataset = apply_smote(train_dataset)

            # 应用SMOTE到测试集（注意：通常不推荐对测试集应用SMOTE，但根据需求这里进行了）
            # test_dataset = test_dataset

            # 构建 DataLoader
            train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

            # 拷贝 meta_model，准备在该任务上做一次内循环训练
            model = copy.deepcopy(sita_p).to(device)
            model.train()

            # (1.1) 在训练集上做若干步训练 (此处只写单 epoch 遍历，你也可多次遍历 train_loader)
            train_loss_sum = 0.0
            for update_step in range(N_U):
                for inputs, labels in train_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device).long().view(-1)

                    # 清空梯度
                    for para in model.parameters():
                        para.grad = None

                    outputs = model(inputs)
                    loss = criterion(outputs, labels.float().view(-1, 1))

                    # 计算正则化项: 0.5 * lambda_reg * ||theta_p - theta_g||^2
                    l2_reg = 0.5 * lambda_reg * sum(
                        (p_p - p_g).pow(2).sum() for p_p, p_g in zip(model.parameters(), sita_g.parameters())
                    )

                    total_loss = loss + l2_reg
                    total_loss.backward()

                    # 用固定 alpha 来更新参数 (FO-MAML 内层)
                    grads = [p.grad for p in model.parameters()]
                    sgd_optimize(model.parameters(), fixed_alpha, grads)

                    # 仅累计分类损失，不包括正则化损失
                    train_loss_sum += loss.item()

            if len(train_loader) > 0:
                print(f"Train_task:{i}, Loss per batch:{train_loss_sum / len(train_loader):.4f}")
            else:
                print(f"Train_task:{i}, (train_loader is empty).")

            # (1.2) 使用更新后的 model 在测试集上计算 loss，并对 sita_p 的初始参数做一阶梯度
            model.eval()

            if len(test_loader) > 0:
                # 再 forward 一下整份测试集 loss
                test_loss_sum = 0.0
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device).long().view(-1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.float().view(-1, 1))
                    test_loss_sum += loss.item()

                print(f"Test_task:{i}, Loss per batch:{test_loss_sum / len(test_loader):.4f}")

                # ---- FO-MAML 核心：对“更新后参数”做一次一阶 backward ----
                # 重新取一次 test_loader 里的数据（这里简单地取第一个 batch）做 backward
                inputs_test, labels_test = next(iter(test_loader))
                inputs_test = inputs_test.to(device)
                labels_test = labels_test.to(device).long().view(-1)

                # 切回 train 模式以允许梯度
                model.train()
                for para in model.parameters():
                    para.grad = None

                outputs_test = model(inputs_test)
                loss_test = criterion(outputs_test, labels_test.float().view(-1, 1))

                # 计算正则化项: 0.5 * lambda_reg * ||theta_p - theta_g||^2
                l2_reg = 0.5 * lambda_reg * sum(
                    (p_p - p_g).pow(2).sum() for p_p, p_g in zip(model.parameters(), sita_g.parameters())
                )

                # 总损失用于元更新
                total_loss = loss_test + l2_reg
                total_loss.backward()

                # model.parameters() 里 param.grad 就是一阶近似，用来更新 sita_p
                grads_now = [p.grad.clone() for p in model.parameters()]
                if ini_sita_grad is None:
                    ini_sita_grad = grads_now
                else:
                    ini_sita_grad = inisitagrad_add(ini_sita_grad, grads_now)

                if (i+1) % meta_batchsize == 0 or (i+1)==len(train_tasks):
                    # (1.3) 根据 meta_batchsize 进行一次 meta update

                    # 更新 sita_p 的初始参数
                    if ini_sita_grad is not None:
                        for param, g in zip(sita_p.parameters(), ini_sita_grad):
                            param.grad = g
                        p_optimizer.step()
                        # 清理
                        p_optimizer.zero_grad()
                        step += 1
                        print(f"  -> Meta update step {step} done.")
                        print('--------------------------------------------')

        # 2) 每个 epoch 结束，做一次测试任务的评估
        print('--------------------------------------------')
        print('Test on unseen tasks:')
        total_TP = total_FP = total_TN = total_FN = 0.0
        total_test_loss = 0.0
        test_tasks = tasks_dataset[len(tasks_dataset) - num_test_task:]
        for j, test_task_dataset in enumerate(test_tasks):
            test_idx = j + len(train_tasks)

            try:
                train_dataset, test_dataset = train_test_split(
                    test_task_dataset, test_size=0.3, random_state=0, stratify=test_task_dataset.y_data
                )
            except Exception as e:
                print(f"[Warning] Splitting test task {test_idx} failed: {e}, skip.")
                continue

            # 应用SMOTE到训练集
            train_dataset = apply_smote(train_dataset)

            # # 应用SMOTE到测试集（注意：通常不推荐对测试集应用SMOTE，但根据需求这里进行了）
            # test_dataset = apply_smote(test_dataset)

            train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

            # 复制 meta_model
            test_model = copy.deepcopy(sita_p).to(device)
            test_model.train()

            # 先在当前测试任务的训练集上做一次内循环更新 (FO-MAML 内层训练)
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long().view(-1)
                for p in test_model.parameters():
                    p.grad = None

                out = test_model(inputs)
                loss = criterion(out, labels.float().view(-1, 1))
                loss.backward()

                sgd_optimize(test_model.parameters(),
                             fixed_alpha,
                             [p.grad for p in test_model.parameters()])

            # 再在测试集上评估
            test_model.eval()
            TP = FP = TN = FN = 0.0
            test_loss_sum = 0.0
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long().view(-1)

                out = test_model(inputs)
                loss = criterion(out, labels.float().view(-1, 1))
                test_loss_sum += loss.item()

                # pred: >=0.5 -> 1, else -> 0
                pred = (out >= 0.5).long().view(-1)
                labels = labels.view(-1)  # 确保 shape=[N], 里面是0或1
                for pre, truth in zip(pred, labels):
                    if pre.item() == 1:
                        if truth.item() == 1:
                            TP += 1
                            total_TP += 1
                        else:
                            FP += 1
                            total_FP += 1
                    else:
                        if truth.item() == 0:
                            TN += 1
                            total_TN += 1
                        else:
                            FN += 1
                            total_FN += 1

            if len(test_loader) > 0:
                test_loss_avg = test_loss_sum / len(test_loader)
            else:
                test_loss_avg = 0.0

            total_test_loss += test_loss_avg

            # 打印指标
            total_num = TP + FP + TN + FN
            acc = (TP + TN) / total_num if total_num else 0
            recall = TP / (TP + FN) if (TP + FN) else 0
            precision = TP / (TP + FP) if (TP + FP) else 0
            f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0
            FPR = FP / (FP + TN) if (FP + TN) else 0
            FNR = FN / (TP + FN) if (TP + FN) else 0

            print(f"[Test Task:{test_idx}] Loss:{test_loss_avg:.4f}, "
                  f"Acc:{acc:.4f}, Recall:{recall:.4f}, Precision:{precision:.4f}, F1:{f1:.4f}, "
                  f"FPR:{FPR:.4f}, FNR:{FNR:.4f}")
            print('--------------------------------------------')

        avg_test_loss = total_test_loss / num_test_task if num_test_task else 0
        total_num = total_TP + total_FP + total_TN + total_FN
        acc = (total_TP + total_TN) / total_num if total_num else 0
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) else 0
        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) else 0
        f1 = 2 * total_TP / (2 * total_TP + total_FP + total_FN) if (2 * total_TP + total_FP + total_FN) else 0
        FPR = total_FP / (total_FP + total_TN) if (total_FP + total_TN) else 0
        FNR = total_FN / (total_TP + total_FN) if (total_TP + total_FN) else 0

        print(f"Epoch:{epoch}, the loss per test task:{avg_test_loss:.4f},"
              f"Acc:{acc:.4f}, Recall:{recall:.4f}, Precision:{precision:.4f}, F1:{f1:.4f},"
              f"FPR:{FPR:.4f}, FNR:{FNR:.4f}")
        print(f"TP:{total_TP}, FP:{total_FP}, TN:{total_TN}, FN:{total_FN}")

        print('--------------------------------------------\n')

        # 可选：保存模型
        torch.save(sita_p, f'{dataset_path}/batch_size_{meta_batchsize}/LWso_sita_p.pth')

    file.close()
