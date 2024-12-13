import torch
import torch.optim as optim
import torch.nn as nn
from mlp_model import GaussianPredictorMLP
import os
def train_mlp(data_path="data/train_data.pt", save_path="models/mlp_model.pt", num_epochs=1000, lr=0.001):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load data
    data = torch.load(data_path)
    inputs, labels = data['inputs'].cuda(), data['labels'].cuda()

    # Initialize model, loss, and optimizer
    model = GaussianPredictorMLP().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

    # Save trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Run the training
if __name__ == "__main__":
    train_mlp()