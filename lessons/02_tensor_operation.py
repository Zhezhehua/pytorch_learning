import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

print(x + y)
print(x * y)

a = torch.tensor([[1, 2],
                  [3, 4]])

b = torch.tensor([[5, 6],
                  [7, 8]])

print(a @ b)
print(x @ y)

print(a.shape)
print(b.shape)
print(a.dtype)