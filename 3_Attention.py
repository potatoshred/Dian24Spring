import torch
from torch import nn
from torch.nn import functional as F

def transpose_qkv(X, n_heads):
    # Q, K, V: (batch_size, n_qkv, n_hidden)
    X = X.reshape(X.shape[0], X.shape[1], n_heads, -1).transpose(1,2)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transback_qkv(X, n_heads):
    X = X.reshape(-1, n_heads, X.shape[1], X.shape[2]).transpose(1,2)
    return X.contiguous().reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_hidden, n_heads):
        super().__init__()

        self.n_hidden = n_hidden
        self.n_heads = n_heads

        self.Wq = nn.Linear(n_hidden, n_hidden, bias=False)
        self.Wk = nn.Linear(n_hidden, n_hidden, bias=False)
        self.Wv = nn.Linear(n_hidden, n_hidden, bias=False)

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

        # 计算注意力权重
        A = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.n_hidden // self.n_heads))
        if mask is not None:
            A = A.masked_fill(mask == 0, -1e9)
        A = F.softmax(A, dim=-1)

        # 计算注意力层的输出
        O = torch.matmul(A, V)

        # 变回(batch_size, n_qkv, n_hidden)
        O = transback_qkv(O, self.n_heads)

        # 投影输出
        O = self.Wo(O)

        return A, O


# 创建随机矩阵
batch_size = 2
n_qkv = 10
n_hidden = 32
n_heads = 8

Q = torch.randn(batch_size, n_qkv, n_hidden)
K = V = torch.randn(batch_size, n_qkv*2, n_hidden)

# 创建多头注意力模型
attention = MultiHeadAttention(n_hidden, n_heads)

# 使用模型进行前向传播
output = attention(Q, K, V)

# 提取注意力权重
attention_weights, _ = output

print(attention_weights)