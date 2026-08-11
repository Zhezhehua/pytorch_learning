import torch
import torch.nn as nn

# 创建模型
linear = nn.Linear(3, 1)

# 输入
x = torch.tensor([[1.0, 2.0, 3.0]])

# 真实答案
target = torch.tensor([[10.0]])

# 模型预测
prediction = linear(x)


print("prediction:")
print(prediction)


# 创建损失函数
loss_fn = nn.MSELoss() #均方误差(MSE=​(∑(prediction−target)^2)/n


# 计算损失
loss = loss_fn(prediction, target)


print("loss:")
print(loss)


# 反向传播(03)
loss.backward() 
'''根据 Loss 反向计算所有需要学习参数的梯度'''

print("weight gradient:")
print(linear.weight.grad)

print("bias gradient:")
print(linear.bias.grad)