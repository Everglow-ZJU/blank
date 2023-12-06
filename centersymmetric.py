import torch
import torch.nn as nn
torch.manual_seed(42)

class SymmetricMatrixLayer_2(nn.Module):
    def __init__(self, size):
        super(SymmetricMatrixLayer_2, self).__init__()
        # 初始化上三角向量
        self.size=size
        self.upper_tri_vector = nn.Parameter(torch.randn(size*(size+1)//2))
        self.mask = torch.triu(torch.ones(size, size), diagonal=0).bool().cuda() #todo:is device right

    def forward(self):
        # 从上三角向量构造出完整的对称矩阵
        symmetric_matrix = torch.zeros(self.mask.size(), device=self.upper_tri_vector.device)
        symmetric_matrix[self.mask] = self.upper_tri_vector
        symmetric_matrix=symmetric_matrix+symmetric_matrix.t()-torch.diag(symmetric_matrix.diag())
        return symmetric_matrix

# 使用示例
size = 4
symmetric_layer = SymmetricMatrixLayer_2(size).to('cuda')
print(symmetric_layer.upper_tri_vector)
symmetric_matrix = symmetric_layer()
print(symmetric_matrix)
