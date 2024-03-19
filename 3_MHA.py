import torch
from torch import nn
from torch.nn import functional as F


def transpose_qkv(X, n_heads):
    # Q, K, V: (batch_size, n_qkv, n_hidden)
    X = X.reshape(X.shape[0], X.shape[1], n_heads, -1).transpose(1, 2)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transback_qkv(X, n_heads):
    X = X.reshape(-1, n_heads, X.shape[1], X.shape[2]).transpose(1, 2)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_q, d_k, d_v, n_hidden, n_heads):
        super().__init__()

        self.n_hidden = n_hidden
        self.n_heads = n_heads

        self.Wq = nn.Linear(d_q, n_hidden, bias=False) # Wqk和Wv的维度可以更自由一些，这里简便些全部统一了
        self.Wk = nn.Linear(d_k, n_hidden, bias=False)
        self.Wv = nn.Linear(d_v, n_hidden, bias=False)

        self.Wo = nn.Linear(n_hidden, n_hidden, bias=False)

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch_size, n_qkv, n_hidden)

        # 计算Q,K,V投影
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        # (batch_size, n_heads, n_qkv, n_hidden // n_heads)
        Q = transpose_qkv(Q, self.n_heads)
        K = transpose_qkv(K, self.n_heads)
        V = transpose_qkv(V, self.n_heads)

        # 计算注意力权重： Q和K的特征维度相同，可采用缩放点积法
        A = torch.matmul(Q, K.transpose(-2, -1)) / \
            torch.sqrt(torch.tensor(self.n_hidden // self.n_heads))
        if mask is not None:
            A = A.masked_fill(mask == 0, -1e9)
        A = nn.Softmax(dim=-1)(A)

        # 计算注意力层的输出
        O = torch.matmul(A, V)

        # 变回(batch_size, n_qkv, n_hidden)
        O = transback_qkv(O, self.n_heads)

        # 投影输出
        O = self.Wo(O)

        return A, O



# 创建随机矩阵
batch_size = 2
n_q = 10
n_kv = 20
n_hidden = 32
n_heads = 4
# qkv特征维度
d_q = 4
d_k = 9
d_v = 7

Q = torch.randn(batch_size, n_q, d_q)
K = torch.randn(batch_size, n_kv, d_k)
V = torch.randn(batch_size, n_kv, d_v)

# 创建多头注意力模型
attention = MultiHeadAttention(d_q, d_k, d_v, n_hidden, n_heads)

# 使用模型进行前向传播
output = attention(Q, K, V)

# 提取注意力权重
attention_weights, _ = output

print(attention_weights)
