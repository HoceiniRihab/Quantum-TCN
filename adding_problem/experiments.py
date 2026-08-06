import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime
import time

import QTCN
from utils import *

def train_qtcn_with_test_tracking(model, train_loader, test_loader, criterion, optimizer, num_epochs=100, test_every=5, save_path="qtcn_results.pkl"):
    model.train()
    train_losses = []
    test_losses = []
    test_accuracies = []
    test_epochs = []
    
    for epoch in range(num_epochs):
       #Training
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            #  for QTCN: [batch, seq_length, features]
            data = data.permute(0, 2, 1) 
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Testing
        if epoch % test_every == 0 or epoch == num_epochs - 1:
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    
                    data = data.permute(0, 2, 1)  
                    output = model(data)
                    test_loss += criterion(output, target).item()
                    pred_error = torch.abs(output - target)
                    correct += (pred_error < 0.04).sum().item()  
                    total += target.size(0)
            
            avg_test_loss = test_loss / len(test_loader)
            accuracy = 100. * correct / total
            
            test_losses.append(avg_test_loss)
            test_accuracies.append(accuracy)
            test_epochs.append(epoch)
            
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}, Test Acc: {accuracy:.2f}%')
            
            
            results = {
                'train_losses': train_losses,
                'test_losses': test_losses,
                'test_accuracies': test_accuracies,
                'test_epochs': test_epochs,
                'epoch': epoch,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(results, f)
        
        elif epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {avg_train_loss:.6f}')
    
    return train_losses, test_losses, test_accuracies, test_epochs

# experiment for a specific T value
def run_single_experiment(T, num_epochs=100, hidden_dim=4):
    
    
    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT FOR T = {T}")
    print(f"{'='*60}")
    seq_length = T
    batch_size = 32
    learning_rate = 0.001
    train_size = 1000
    test_size = 500
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"qtcn_T{T}_results_{timestamp}.pkl"
    
    try:
        
        train_X, train_Y = data_generator(train_size, seq_length)
        train_dataset = torch.utils.data.TensorDataset(train_X, train_Y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_X, test_Y = data_generator(test_size, seq_length)
        test_dataset = torch.utils.data.TensorDataset(test_X, test_Y)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        model = QTCN(seq_length=seq_length, input_features=2, hidden_dim=hidden_dim, output_dim=1)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"QTCN model has {total_params} parameters")
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        train_losses, test_losses, test_accuracies, test_epochs = train_qtcn_with_test_tracking(
            model, train_loader, test_loader, criterion, optimizer, num_epochs, test_every=5, save_path=save_path
        )
        
        if train_losses is None:
            print("Training failed!")
            return None
        
        print(f"\nFinal Results for T={T}:")
        print(f"Final Test Loss: {test_losses[-1]:.6f}")
        print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
        print(f"Best Test Loss: {min(test_losses):.6f}")
        print(f"Best Test Accuracy: {max(test_accuracies):.2f}%")
        results = {
            'T': T,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'test_accuracies': test_accuracies,
            'test_epochs': test_epochs,
            'final_test_loss': test_losses[-1],
            'final_test_accuracy': test_accuracies[-1],
            'best_test_loss': min(test_losses),
            'best_test_accuracy': max(test_accuracies),
            'total_params': total_params,
            'save_path': save_path
        }
        
        return results
        
    except Exception as e:
        print(f"ERROR in experiment for T={T}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_all_experiments():
    """Run experiments for T = 200, 400, 600"""
    
    T_values = [200, 400, 600]
    all_results = {}
    
    print("Starting QTCN Adding Problem Experiments")
    print("Testing with 3 Residual Blocks")
    print(f"T values: {T_values}")
    
    for T in T_values:
        try:
            print(f"\n\nStarting experiment for T={T}")
            results = run_single_experiment(T, num_epochs=100, hidden_dim=4)
            
            if results:
                all_results[T] = results
                print(f"Experiment for T={T} completed successfully")
            else:
                print(f"Experiment for T={T} failed")
                
        except Exception as e:
            print(f"Error running experiment for T={T}: {str(e)}")
            continue

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results_path = f"qtcn_all_results_{timestamp}.pkl"
    
    try:
        with open(all_results_path, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"\nAll results saved to {all_results_path}")
    except Exception as e:
        print(f"Error saving all results: {str(e)}")
    
    return all_results

if __name__ == "__main__":
    print("adding problem experiments ")
    print("=" * 50)
    
    results = run_all_experiments()
    
    if results:
        print(f"\nCompleted {len(results)} experiments successfully!")
        for T, result in results.items():
            print(f"T={T}: Final test loss = {result['final_test_loss']:.6f}, Final test accuracy = {result['final_test_accuracy']:.2f}%")
    else:
        print("\nExperiments failed")