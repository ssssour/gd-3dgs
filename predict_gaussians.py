import torch
import numpy as np
from mlp_model import GaussianPredictorMLP

class PointCloud:
    """A simple class to represent a point cloud with only points (no colors)."""
    def __init__(self, points):
        self.points = points

def load_model(model_path="models/mlp_model.pt"):
    # Adjust input_dim to 3 since we're only using xyz
    model = GaussianPredictorMLP(input_dim=3)
    model.load_state_dict(torch.load(model_path))
    model.eval().cuda()
    return model

def predict_gaussians(model, pcd):
    # Prepare inputs from the new PCD (only points, no colors)
    inputs = pcd.points.astype(np.float32)
    inputs_tensor = torch.tensor(inputs).cuda()

    # Run predictions
    with torch.no_grad():
        predictions = model(inputs_tensor).cpu().numpy()
    return predictions

# Example usage:
if __name__ == "__main__":
    model = load_model()

    # Generate random test point cloud data with only xyz
    num_points = 100  # Set number of random points
    points = np.random.rand(num_points, 3)  # Random (x, y, z) positions

    # Create a synthetic PointCloud object
    random_pcd = PointCloud(points=points)

    # Run predictions
    predictions = predict_gaussians(model, random_pcd)
    print(predictions)