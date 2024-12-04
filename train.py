import os
import math
import torch
from torch.optim import Adam

def train(model, train_loader, num_epochs=1, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = Adam(model.parameters(), lr, (0.9, 0.95), weight_decay=0.05)
    
    
    
    for epoch in range(num_epochs):
        for batch_idx, (data) in enumerate(train_loader):
            data = data.to(device)

            optimizer.zero_grad()
            loss, pred = model(data)
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")
    
    warmup_epoch = 20
    lr_func = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / num_epochs * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)
    
    for epoch in range(num_epochs):
        for batch_idx, (data) in enumerate(train_loader):
            data = data.to(device)

            optimizer.zero_grad()
            loss, pred = model(data)
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")
        path = os.path.join(".", f'epoch_{epoch}.pt')
        lr_scheduler.step()
        torch.save(model.state_dict(), path)
        #with open(folder_logs, 'a+') as f:
        #    f.writelines(f"Epoch: {epoch}/{num_epochs}, Loss: {loss.item()} \n")
    return model