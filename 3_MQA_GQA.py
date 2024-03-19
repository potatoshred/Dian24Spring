import torch
import torch.nn as nn

def multi_query_attention(Q, K, V):
    # 注意力权重矩阵
    attention_weights = torch.zeros(Q.size(0), Q.size(1), K.size(1))

    for i in range(Q.size(0)):
        # 计算当前查询与所有键的点积
        dot_products = torch.matmul(Q[i], K.transpose(1, 2))

        # 将点积转换为注意力权重
        attention_weights[i] = torch.softmax(dot_products, dim=-1)

    # 计算加权值
    O = torch.matmul(attention_weights, V)

    return O


def group_query_attention(Q, K, V, num_groups):
    # 查询、键和值投影
    # (b, qkv, 1, d)
    Q = Q.unsqueeze(2)
    K = K.unsqueeze(2)
    V = V.unsqueeze(2)

    # 分组查询
    Q = Q.reshape(Q.size(0), num_groups, -1, Q.size(-1))

    # 初始化注意力权重列表
    attention_weights = []

    # 对每个组应用多查询注意力
    for group in Q:
        attention_weights.append(multi_query_attention(group, K, V))

    # 连接注意力权重
    attention_weights = torch.cat(attention_weights, dim=1)

    # 计算加权值
    attended_values = torch.matmul(attention_weights, V)

    # 连接加权值
    O = attended_values.reshape(attended_values.size(0), -1, attended_values.size(-1))

    return O