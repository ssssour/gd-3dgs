import torch
import torch.optim as optim
import torch.nn as nn
from mlp_model import GaussianPredictorMLP
import os
import glob

def train_mlp(data_path="data/train_data.pt", save_path="models/mlp_model.pt", num_epochs=5000, lr=0.001):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Load data
    data = torch.load(data_path)
    inputs, labels = data['inputs'].cuda(), data['labels'].cuda()
    print(inputs.shape)

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
        if epoch % 10000 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

    # Save trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def train_all_datasets(data_dir="data", output_dir="models", num_epochs=100000, lr=0.001):
    """
    Train a model for each dataset in the data directory and save each model in a unique folder.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all training data files
    data_files = glob.glob(os.path.join(data_dir, "*.pt"))

    if not data_files:
        print("No training data files found.")
        return

    for data_file in data_files:
        # Extract scene name from the file name
        scene_name = os.path.splitext(os.path.basename(data_file))[0]

        # Create a subdirectory for the scene
        scene_dir = os.path.join(output_dir, scene_name)
        os.makedirs(scene_dir, exist_ok=True)

        # Define the save path for the model
        save_path = os.path.join(scene_dir, "mlp_model.pt")

        # Skip training if the model already exists
        if os.path.exists(save_path):
            print(f"Model already exists for scene: {scene_name}, skipping training.")
            continue

        print(f"Training on dataset: {scene_name}")
        train_mlp(data_path=data_file, save_path=save_path, num_epochs=num_epochs, lr=lr)

# Example: Running the training
if __name__ == "__main__":
    train_all_datasets(data_dir="/root/autodl-tmp/gaussian-splatting2/processed_data_upsample", output_dir="models_upsample", num_epochs=100000, lr=0.001)