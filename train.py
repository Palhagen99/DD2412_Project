import os
import math
import torch
import torch.nn as nn
from torch.optim import Adam

def train(model, train_loader, num_epochs=1, lr=1e-4, name="DEFAULT"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = Adam(model.parameters(), lr, (0.9, 0.95), weight_decay=0.05)
    criterion = nn.MSELoss()
    warmup_epoch = 5
    lr_func = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / num_epochs * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)
    
    for epoch in range(num_epochs):
        for batch_idx, (data) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            loss, pred = model(data)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")
        if epoch == num_epochs:
            path = os.path.join(".", f'{name}.pt')
            torch.save(model.state_dict(), path)
        lr_scheduler.step()
        
    return model