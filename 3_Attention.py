import torch
from torch import nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.Wo = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch_size, seq_len, d_model)

        # 计算Q,K,V投影。
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        # 将投影重塑为 (batch_size, n_heads, seq_len, d_model // n_heads)。
        Q = Q.view(Q.shape[0], Q.shape[1], self.n_heads, -1)
        K = K.view(K.shape[0], K.shape[1], self.n_heads, -1)
        V = V.view(V.shape[0], V.shape[1], self.n_heads, -1)

        # 计算注意力权重。
        A = torch.matmul(Q, K.transpose(2, 3)) / torch.sqrt(torch.tensor(self.d_model // self.n_heads))
        if mask is not None:
            A = A.masked_fill(mask == 0, -1e9)
        A = F.softmax(A, dim=-1)

        # 计算注意力层的输出。
        O = torch.matmul(A, V)

        # 将输出重塑回 (batch_size, seq_len, d_model)。
        O = O.view(O.shape[0], O.shape[1], -1)

        # 投影输出。
        O = self.Wo(O)

        return A, O


# 创建随机矩阵
batch_size = 1
seq_len = 10
d_model = 256
n_heads = 8

random_input = torch.randn(batch_size, seq_len, d_model)

# 创建多头注意力模型
attention = MultiHeadAttention(d_model, n_heads)

# 使用模型进行前向传播
output = attention(random_input, random_input, random_input)

# 提取注意力权重
attention_weights, _ = output

print(attention_weights)