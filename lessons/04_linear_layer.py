import torch
import torch.nn as nn


# 输入维度 3
# 输出维度 2
linear = nn.Linear(3, 2)
'''输入三个特征，输出两个结果'''

x = torch.tensor([
    [1.0, 2.0, 3.0]
])


y = linear(x)


print(y)
print(y.shape)

print(linear.weight)
print(linear.bias)

'''y=xWT+b,即w为2x3矩阵'''