import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_dl(model, device, train_dataset, epochs, batch_size, lr, output_path, use_ipex: bool = False):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    if use_ipex and device != "cuda":
        try:
            import intel_extension_for_pytorch as ipex

            model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)
            model = torch.compile(model, backend="ipex") # 实验性功能
        except:
            print("ipex uncapable!")
    best_loss = float("inf")
    for epoch in tqdm(range(epochs), desc="Training", colour="blue"):
        loss = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), os.path.join(output_path, "model.pth"))
                
            loss.backward()
            optimizer.step()
            
    
    return best_loss


def predict(model, device, test_dataset, batch_size, output_path, use_ipex: bool = False):
    model.eval()
    if use_ipex and device != "cuda":
        try:
            import intel_extension_for_pytorch as ipex

            model = ipex.optimize(model)
            model = torch.compile(model, backend="ipex") # 实验性功能
        except:
            print("ipex uncapable!")
    
    model.load_state_dict(torch.load(os.path.join(output_path, "model.pth")))
    model.to(device)
    loader = DataLoader(test_dataset, batch_size=batch_size)

    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy
    