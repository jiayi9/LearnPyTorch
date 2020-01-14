import torch

x = torch.ones(2, 2, requires_grad=True)
x

y = x + 2
y

print(y.grad_fn)

z = y * y * 3

out = z.mean()

print(x)
print(y)
print(z)
print(out)

#tensor([[1., 1.],
#        [1., 1.]], requires_grad=True)     
#tensor([[3., 3.],
#        [3., 3.]], grad_fn=<AddBackward0>)   y = x + 2
#tensor([[27., 27.],
#        [27., 27.]], grad_fn=<MulBackward0>)   z = y * y * 3
#tensor(27., grad_fn=<MeanBackward1>)         out = z.mean()

# x -> y -> z -> out

out.backward()

x.grad

y.grad


#a = torch.randn(2, 2)
#
#a = ((a * 3) / (a - 1))
#
#print(a.requires_grad)
#a.requires_grad_(True)
#print(a.requires_grad)
#
#b = (a * a).sum()
#
#print(b.grad_fn)




# multiply element by element
x = torch.tensor([5, 3, 4])
y = torch.tensor([5, 3, 9])
x*y
