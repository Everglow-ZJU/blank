import torch
import torch.nn as nn

class SymmetricMatrixLayer(nn.Module):
    def __init__(self, size):
        super(SymmetricMatrixLayer, self).__init__()
        self.upper_tri_values = nn.Parameter(torch.randn(size, size))
        self.mask = torch.triu(torch.ones(size, size), diagonal=0).bool()

    def forward(self):
        upper_tri = self.upper_tri_values.masked_select(self.mask).view(-1)
        symmetric_matrix = torch.zeros(self.upper_tri_values.size(), device=self.upper_tri_values.device)
        symmetric_matrix[self.mask] = upper_tri
        symmetric_matrix[self.mask.transpose(0, 1)] = upper_tri
        return symmetric_matrix

# 使用示例
size = 4
symmetric_layer = SymmetricMatrixLayer(size)
symmetric_matrix = symmetric_layer()
print(symmetric_matrix)
