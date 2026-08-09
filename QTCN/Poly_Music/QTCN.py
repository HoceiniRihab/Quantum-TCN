import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np


class QuantumDilatedConvolution(nn.Module):
    
    def __init__(self, n_qubits, dilation_rate):
        super().__init__()
        self.n_qubits = n_qubits
        self.dilation_rate = min(dilation_rate, n_qubits - 1)
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev, interface='torch')
        def circuit(inputs, weights):
            # 2 layers of encoding
            for i in range(n_qubits):
                temporal_phase = inputs[i] * np.pi * (2*i + 1) / n_qubits
                qml.PhaseShift(temporal_phase, wires=i)
            
            for i in range(n_qubits):
                qml.RY(inputs[i] * np.pi, wires=i)
            
            weight_idx = 0
            for i in range(n_qubits - self.dilation_rate):
                if i % 2 == 0 and weight_idx < len(weights):  
                    qml.CNOT(wires=[i, i + self.dilation_rate])
                    qml.RZ(weights[weight_idx], wires=i + self.dilation_rate)
                    weight_idx += 1
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        self.qnode = circuit
        self.weight_shape = (max(1, n_qubits // 4),)  
        self.weights = nn.Parameter(0.01 * torch.randn(self.weight_shape))

    def forward(self, x):

        if x.dim() != 3:
            raise ValueError(f"Input tensor must have 3 dimensions, got {x.dim()} dimensions")
            
        batch_size, channels, sequence_length = x.shape
        output = torch.zeros((batch_size, self.n_qubits, sequence_length), 
                            device=x.device, dtype=x.dtype)
        weights_np = self.weights.detach().cpu().numpy()
        
        for j in range(sequence_length):
            
            x_slice = x[:, :, j]
            for i in range(batch_size):
                input_data = x_slice[i, :min(channels, self.n_qubits)].detach().cpu().numpy()
                if len(input_data) < self.n_qubits:
                    input_data = np.pad(input_data, 
                                    (0, self.n_qubits - len(input_data)), 
                                    'constant', constant_values=0)
                
                circuit_output = self.qnode(input_data, weights_np)
                output[i, :, j] = torch.tensor(circuit_output, device=x.device)
        
        return output


class Chomp1d(nn.Module):
    
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class InputLayer(nn.Module):
    def __init__(self, seq_length, hidden_dim):
        super(InputLayer, self).__init__()
        internal_dim = max(2, hidden_dim // 4)
        self.linear = nn.Linear(2, internal_dim)
        self.projection = nn.Linear(internal_dim, hidden_dim)
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        if x.dim() == 2:
            x = x.unsqueeze(1).repeat(1, self.seq_length, 1)
            
        x = F.relu(self.linear(x))
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        
        # bn
        x = self.bn(x)
        
        return x


class FullyConnected(nn.Module):
    
    def __init__(self, input_size, hidden_size=8, output_size=1, dropout_prob=0.2):
        super(FullyConnected, self).__init__()
        self.input_size = input_size
        self.register_buffer('padding_mask', torch.ones(input_size, dtype=torch.bool))
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1).reshape(batch_size, -1)
        if x.shape[1] < self.input_size:
            pad_size = self.input_size - x.shape[1]
            x = F.pad(x, (0, pad_size), 'constant', 0)
        elif x.shape[1] > self.input_size:
            x = x[:, :self.input_size]
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ResidualBlock(nn.Module):
    
    def __init__(self, channels, dilation_rate, kernel_size=3, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.dilation_rate = dilation_rate
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) * dilation_rate
        self.qdcnn1 = QuantumDilatedConvolution(
            n_qubits=channels,
            dilation_rate=dilation_rate
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.qdcnn2 = QuantumDilatedConvolution(
            n_qubits=channels,
            dilation_rate=dilation_rate
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(channels)
        
        
        self.skip_connection = nn.Conv1d(channels, channels, kernel_size=1, padding=0)
        self.final_relu = nn.ReLU()

    def forward(self, x):
        residual = self.skip_connection(x)
        
        # Main path: quantum dilated Conv → Chomp1d → ReLU → Dropout → quantum dilated Conv → Chomp1d → ReLU → Dropout
        out = self.qdcnn1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.qdcnn2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.bn(out)
        

        
        if out.shape != residual.shape:
            residual = F.pad(residual, (0, out.shape[2] - residual.shape[2]))
            
        # Residual connection
        return self.final_relu(out + residual)


class QTCN(nn.Module):
    def __init__(self, seq_length=32, input_features=2, hidden_dim=6, output_dim=1):
        super(QTCN, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.input_layer = InputLayer(seq_length, hidden_dim)
        self.residual_block1 = ResidualBlock(hidden_dim, dilation_rate=1)
        self.residual_block2 = ResidualBlock(hidden_dim, dilation_rate=2)
        self.residual_block3 = ResidualBlock(hidden_dim, dilation_rate=4)
        fc_input_size = 4800  
        self.fc = FullyConnected(
            input_size=fc_input_size, 
            hidden_size=8, 
            output_size=output_dim
        )

    def forward(self, x):
    
        
        x = self.input_layer(x)  # [batch, hidden_dim, seq_length]
        x = self.residual_block1(x)  # dilation_rate=1
        x = self.residual_block2(x)  # dilation_rate=2
        x = self.residual_block3(x)  # dilation_rate=4
        x = self.fc(x)
        
        return x