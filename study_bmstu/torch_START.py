import torch

x = torch.tensor([-2.0],requires_grad = True )
y = torch.tensor(10.0, requires_grad = True)

z = x**2 + y **2

z.backward()

print(z)

grad_x = x.grad
grad_y = y.grad

print(grad_x,grad_y)

x_new = x.item()
y_new = y.item()


x = torch.tensor([3.0], requires_grad=True)
f = x**2
f_detached = f.detach()
f_detached.backward()

