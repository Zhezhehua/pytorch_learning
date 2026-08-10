'''生成3x3随机矩阵
import torch
x = torch.randn(3, 3)
print(x)
device = torch.device("cuda")
x = x.to(device)
print(x)'''
'''体验 GPU 矩阵计算'''
import torch
device = torch.device("cuda")
x = torch.randn(10000,10000)
x = x.to(device)
y = x @ x
print(y.device)
#神经网络训练本质:阵乘法+梯度计算+参数更新