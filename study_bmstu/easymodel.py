import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

x1 = torch.linspace(1.0,3,1000 ).unsqueeze(1)
x2 = torch.randn(1000, 1)
X = torch.cat((x1,x2),dim = 1 )
noise = torch.randn(x1.size())

y = 3*x1-2*x2+5+noise

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 52 )



model = nn.Linear(2,1)
epochs  = 1000
loss_fn  = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.1)

losses = []
for epoch in range(epochs):
    y_pred = model(x_train)
    loss = loss_fn(y_pred,y_train)
    if epoch > 100:     losses.append(loss.item())


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100==0: print(loss.item())

y_test_pred = model(x_test).detach()
print(x1.shape)
print(noise.shape)
a = model.weight[0][0].item()
a2 = model.weight[0][1].item()
b = model.bias.item()
print(f"Модель выучила: y = {a:.2f}x1 + {a2:.2f}x2 + {b:.2f}")

