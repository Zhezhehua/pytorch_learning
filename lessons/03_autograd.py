'''import torch

#requires_grad=True表示x是自变量
x = torch.tensor(3.0, requires_grad=True)

y = x ** 2 

y.backward()#求导(2x)并保存到x.grad

print(x.grad)'''
#当w为自变量，x为参数时
'''import torch

w = torch.tensor(2.0, requires_grad=True)

x = torch.tensor(3.0)

y = w * x 

y.backward() #求导是x

print(w.grad) #输出x的值'''

import torch
x = torch.tensor(5.0, requires_grad=True)
y = x ** 3
y.backward()
print(x.grad)