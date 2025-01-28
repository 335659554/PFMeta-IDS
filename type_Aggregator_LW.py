import os
import pickle
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import copy
import torch.nn.functional as F
from torch import nn, optim

feature_num = 9
import math

class_num = 2

# --------------------
# 预设一些全局超参数
# --------------------
attack_type = 'dos'
dirichlet_alpha = 100.0
split_size = 512 # 每个客户端的任务集大小
meta_batchsize = 256 # 32,64,128,256
N_U = 1
total_comm_rounds = 800  # 总通信轮数，可根据需要修改
num_clients = 20  # 客户端总数，举例
lambda_reg = 2.0  # 等等... 需要根据您原本想如何聚合而定

# 聚合时使用的学习率或超参数
aggregate_beta = 0.01
weight_decay = 0.01

# 您的客户端脚本名称
CLIENT_SCRIPT = 'type_Client_LWCNN.py'

# 服务器可见的客户端数据/模型目录
# 注意：与客户端脚本中 dataset_path 拼接保持一致
# 假设客户端存储路径为：clients_dataset/{attack_type}/alpha{dirichlet_alpha}/{client}/splitted_dataset/batch_size_{meta_batchsize}
BASE_CLIENT_DATASET_PATH = 'clients_dataset'

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Server] Using device: {device}")


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
        x = x.view(-1, 1, feature_num)
        # (1) 标准卷积 => [N, 32, L]
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


def run_client(client_id, round_idx):
    """
    以子进程方式启动客户端训练脚本。
    注意这里使用的是同步运行(阻塞)，如果需要异步并行，会配合线程池 / 进程池并行处理。
    """
    # 构造命令行参数
    cmd = [
        'python', CLIENT_SCRIPT,
        '--attack_type', str(attack_type),
        '--dirichlet_alpha', str(dirichlet_alpha),
        '--meta_batchsize', str(meta_batchsize),
        '--N_U', str(N_U),
        '--comm_round_n', str(round_idx),
        '--client', str(client_id),
        '--lambda_reg', str(lambda_reg),
        '--split_size', str(split_size)
    ]

    # 启动脚本
    print(f"[Server] Starting training for Client {client_id} (Round {round_idx})...")
    subprocess.run(cmd, check=True)
    # 如果需要在 Windows 下隐藏窗口或其他特殊需求，可做更多定制


def aggregate_gradients_and_update_sita_g(round_idx):
    """
    在所有客户端训练完成后，服务器读取每个客户端的 LWso_sita_g_grad.pkl，
    根据梯度相似度 (公式(6)) + softmax 权重 (公式(7))，对每个客户端做个性化聚合更新。
    最终更新每个客户端本地的 LWso_sita_g.pth。
    """

    # --- 1) 读取每个客户端的 sita_g_grad.pkl ---
    client_grad_list = []
    valid_clients = []
    for client_id in range(num_clients):
        dataset_path = os.path.join(
            BASE_CLIENT_DATASET_PATH,
            attack_type,
            f'alpha{dirichlet_alpha}',
            str(client_id),
            'splitted_dataset',
            f'batch_size_{meta_batchsize}'
        )
        grad_path = os.path.join(dataset_path, 'LWso_sita_g_grad.pkl')
        if os.path.exists(grad_path):
            with open(grad_path, 'rb') as f:
                loaded_grad = pickle.load(f)
                # 确保所有梯度张量都在 GPU 上
                if loaded_grad is None:
                    print(f"[Server][Warning] Grad file for client {client_id} is None. Skip this client.")
                    continue
                grad_data = [g.to(device) for g in loaded_grad]
                # grad_data 是一个 list[Tensor,...]，每个Tensor对应模型某一层/某部分梯度
            client_grad_list.append(grad_data)
            valid_clients.append(client_id)
        else:
            print(f"[Server][Warning] Grad file not found for client {client_id}, skip in aggregation.")

    # 如果没有任何客户端上传了梯度，就跳过
    if not client_grad_list:
        print("[Server][Warning] No gradients found from any client. Skipping aggregation.")
        return

    # --- 2) 为计算相似度，将每个客户端的梯度打平为一个向量 ---
    def flatten_grad(grad_list):
        """ 将 list of Tensors 拼接为一个 1D 向量 """
        return torch.cat([g.reshape(-1) for g in grad_list], dim=0)

    # 打平后的梯度向量列表
    client_grad_vectors = [flatten_grad(g) for g in client_grad_list]
    # 每个梯度向量的 L2 范数 (|Δθ_k^t|)
    client_grad_norms = [v.norm(p=2) for v in client_grad_vectors]

    # --- 3) 依次为每个客户端计算个性化聚合，更新其 sita_g.pth ---
    for idx_k, client_id_k in enumerate(valid_clients):
        # 对于客户端 k:
        #    1) 计算与所有客户端 j 的相似度 sim_theta(k,j)
        #    2) 用 softmax( sim ) 得到权重 (m_theta)_{k,j}
        #    3) 加权求和梯度后，叠加到 (theta_g)_k^t 上

        # (3.1) 读取客户端 k 的当前 sita_g
        dataset_path_k = os.path.join(
            BASE_CLIENT_DATASET_PATH,
            attack_type,
            f'alpha{dirichlet_alpha}',
            str(client_id_k),
            'splitted_dataset',
            f'batch_size_{meta_batchsize}'
        )
        model_path_k = os.path.join(dataset_path_k, 'LWso_sita_g.pth')
        if not os.path.exists(model_path_k):
            print(f"[Server][Warning] Model file not found for client {client_id_k}. Skip update.")
            continue

        client_model_k = torch.load(model_path_k, map_location=device)
        client_model_k.to(device)

        # (3.2) 计算客户端 k 与所有客户端 j 的相似度
        sim_list = []
        grad_k = client_grad_vectors[idx_k]  # flatten后的向量
        norm_k = client_grad_norms[idx_k] + 1e-12  # 防止除零

        for idx_j in range(len(valid_clients)):
            grad_j = client_grad_vectors[idx_j]
            norm_j = client_grad_norms[idx_j] + 1e-12
            # 点积
            dot_kj = torch.dot(grad_k, grad_j)
            # 余弦相似度
            sim_kj = dot_kj / (norm_k * norm_j)
            sim_list.append(sim_kj)

        # (3.3) 使用 softmax(sim_list) 得到权重 (m_theta)_{k,j}
        sim_tensor = torch.stack(sim_list, dim=0)  # [K]
        weight_tensor = F.softmax(sim_tensor, dim=0)  # [K]，其中 j=0..K-1

        # (3.4) 加权求和各客户端 j 的梯度 (未 flatten 的形式)
        #       注意：client_grad_list[j] 里是每一层的梯度(list of Tensors)，逐层相加
        weighted_grad = []
        # 先拿客户端0的grad作模板，创建一个同形状的0张量list
        template_zero = [torch.zeros_like(t) for t in client_grad_list[0]]

        for layer_idx in range(len(template_zero)):
            # layer_idx 表示模型中某一层(或某个Tensor)的梯度
            agg = torch.zeros_like(template_zero[layer_idx])
            # 逐个客户端 j 累加 (m_theta)_{k,j} * Δθ_j^t(layer_idx)
            for idx_j, w_j in enumerate(weight_tensor):
                agg += w_j * client_grad_list[idx_j][layer_idx]
            weighted_grad.append(agg)

        # (3.5) 执行最终更新：
        #       (θ_g)_k^{t+1} = (θ_g)_k^t +  ∑_{j=1}^K (m_theta)_{k,j} * (Δθ_g)_j^t
        #       即在每一层的参数上，加对应层的 weighted_grad
        with torch.no_grad():
            for param, w_grad in zip(client_model_k.parameters(), weighted_grad):
                param.sub_(aggregate_beta * w_grad)  # in-place加上加权后的梯度

        # (3.6) 保存更新后的 model
        torch.save(client_model_k, model_path_k)
        print(f"[Server] Updated sita_g for Client {client_id_k} in Round {round_idx} with personalized gradient.")

    print(f"[Server] Round {round_idx} aggregation finished (personalized).")


def main():
    # 创建结果保存的根目录 (如果不存在)
    os.makedirs(BASE_CLIENT_DATASET_PATH, exist_ok=True)
    # g_path = f'clients_dataset/{attack_type}/alpha{dirichlet_alpha}/LWso_sita_g_b{meta_batchsize}.pth'
    g_path = f'clients_dataset/ini_sita_g.pth'
    if os.path.exists(g_path):
        sita_g = torch.load(g_path, map_location=device)
        print("Loaded existing g_model.")
    else:
        sita_g = LWCNN().to(device)
        torch.save(sita_g, g_path)
        print("Initialized a new g_model.")

    for round_idx in range(total_comm_rounds):
        print(f"\n======================")
        print(f"   Round {round_idx}  ")
        print(f"======================")

        # -------------------
        # 1) 并行启动客户端
        # -------------------
        with ProcessPoolExecutor(max_workers=num_clients) as executor:
            futures = []
            for client_id in range(num_clients):
                # 提交并行任务
                futures.append(executor.submit(run_client, client_id, round_idx))

            # 等待全部完成
            for future in as_completed(futures):
                # 如果您需要获取 return 的结果或捕捉异常，可以在这里处理
                try:
                    future.result()
                except Exception as e:
                    print(f"[Server] A client training failed with exception: {e}")

        # -------------------
        # 2) 全部客户端训练结束后，服务器进行聚合
        # -------------------
        print(f"[Server] All clients finished Round {round_idx}. Start aggregation...")
        aggregate_gradients_and_update_sita_g(round_idx)

        # 这里可选地做一些评估、记录日志等
        print(f"[Server] Round {round_idx} aggregation finished.")

    print("All rounds finished!")


if __name__ == "__main__":
    main()
