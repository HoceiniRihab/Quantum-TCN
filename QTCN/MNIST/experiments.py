import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

import QTCN 
from utils import *


class MNISTAdapter(nn.Module):
    """MNIST images to QTCN input format"""
    
    def __init__(self, qtcn_model):
        super().__init__()
        self.qtcn = qtcn_model
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(784, 64)
        self.classifier = nn.Linear(1, 10)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.flatten(x)
        x = self.projection(x)
        #[batch, seq_length=32, features=2]
        x = x.view(batch_size, 32, 2)
        x = self.qtcn(x)  # [batch, 1]
        x = self.classifier(x) 
        
        return x


def train_epoch(model, train_loader, optimizer, criterion, device):
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        pbar.set_postfix({
            'loss': f'{total_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    
    return test_loss, accuracy


def main():
    
    batch_size = 32
    epochs = 100
    learning_rate = 0.001
    data_root = './data'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print('Loading MNIST dataset')
    train_loader, test_loader = data_generator(data_root, batch_size)
    qtcn = QTCN(seq_length=32, input_features=2, hidden_dim=6, output_dim=1)
    model = MNISTAdapter(qtcn).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                      factor=0.5, patience=2)
    

    best_accuracy = 0
    print('\ntraining.\n')
    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}')
        print('-' * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, 
                                           criterion, device)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step(test_acc)
        print(f'\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%\n')
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'best_qtcn_mnist.pth')
            print(f'✓ New best model saved! (Accuracy: {best_accuracy:.2f}%)\n')
    
    print('=' * 60)
    print(f'Training completed!')
    print(f'Best Test Accuracy: {best_accuracy:.2f}%')
    print('=' * 60)


if __name__ == '__main__':
    main()