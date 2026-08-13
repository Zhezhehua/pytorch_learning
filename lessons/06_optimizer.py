'''optimizer的命题：
得到loss梯度: loss.backward()后，如何改变模型参数'''
import torch
import torch.nn as nn

# 输入
x = torch.tensor([[1.0, 2.0, 3.0]])

# 目标
target = torch.tensor([[10.0]])

# 创建模型
linear = nn.Linear(3, 1)

# 创建优化器
'''SGD随机梯度下降 .parameters()让linear里
的参数(weight,bias)参与学习,lr是学习率'''
optimizer = torch.optim.SGD(
    linear.parameters(),
    lr=0.01
)

'''# 模型预测
prediction = linear(x)

print("prediction:")
print(prediction)

# 损失函数
loss_fn = nn.MSELoss()

# 计算损失
loss = loss_fn(prediction, target)

print("loss:")
print(loss)

# 反向传播
loss.backward()

# 查看梯度
print("weight gradient:")
print(linear.weight.grad)

print("bias gradient:")
print(linear.bias.grad)

# 更新参数
optimizer.step()'''
'''loss.backward()
      ↓
    计算梯度 → optimizer.step()
                    ↓
            修改 weight / bias'''

'''print("after optimizer step:")
print(linear.weight)
print(linear.bias)'''
loss_fn = nn.MSELoss()

for epoch in range(100):

    # 模型预测
    prediction = linear(x)

    # 计算损失
    loss = loss_fn(prediction, target)

    # 清空之前的梯度
    optimizer.zero_grad()

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    # 每10次打印一次
    if epoch % 10 == 0:
        print(f"epoch: {epoch}, loss: {loss.item():.4f}")