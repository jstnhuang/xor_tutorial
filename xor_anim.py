from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class HyperParameters:
    lr: float = 1
    epochs: int = 24


class XorDataset(Dataset):
    def __init__(self) -> None:
        self.data = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.labels = torch.tensor([0, 1, 1, 0])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class XorNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    torch.manual_seed(0)
    train_data = XorDataset()
    data_loader = DataLoader(train_data, batch_size=4, shuffle=True)

    device = "cpu"
    print(f"Using {device} device")

    hp = HyperParameters()
    model = XorNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=hp.lr)

    # --- Animation Setup ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axis_limit = 2.5  # Adjusted axis limit for better visualization

    # Use the full dataset for consistent plotting
    plot_X = train_data.data.to(device)
    plot_y = train_data.labels

    def update(epoch):
        # --- Training Step for one epoch ---
        model.train()
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

        # --- Plotting Step ---
        model.eval()
        with torch.no_grad():
            # Get model outputs at different layers for the full dataset
            tf1 = model.fc1(plot_X)
            relu = F.relu(tf1)
            tf2 = model.fc2(relu)

            correct = (model(plot_X).argmax(1) == plot_y).sum().item()
            current_loss = loss_fn(model(plot_X), plot_y).item()

            print(
                f"Epoch {epoch+1}/{hp.epochs}, loss: {current_loss:>4f}, acc: {correct/4.0}"
            )
            fig.suptitle(
                f"Epoch: {epoch + 1}/{hp.epochs} | Loss: {current_loss:.4f} | Accuracy: {correct/4.0}",
                fontsize=16,
            )

            for i, res in enumerate([tf1, relu, tf2]):
                ax = axes[i]
                ax.clear()  # Clear previous frame's plot

                # Plot data points
                ax.scatter(
                    res[plot_y == 0, 0].numpy(),
                    res[plot_y == 0, 1].numpy(),
                    c="blue",
                    label="0",
                )
                ax.scatter(
                    res[plot_y == 1, 0].numpy(),
                    res[plot_y == 1, 1].numpy(),
                    c="red",
                    label="1",
                )

                # Set titles and labels
                layer_names = ["Layer 1 (Linear)", "Layer 1 (ReLU)", "Layer 2 (Output)"]
                ax.set_title(layer_names[i])
                ax.set_xlabel("x1")
                ax.set_ylabel("x2")
                ax.set_xlim(-axis_limit, axis_limit)
                ax.set_ylim(-axis_limit, axis_limit)
                ax.legend()
                ax.grid(True)
                ax.set_aspect("equal", adjustable="box")
                ax.axhline(0, color="black", linewidth=0.5)
                ax.axvline(0, color="black", linewidth=0.5)

                # Plot decision boundary for the final layer
                if i == 2:
                    line_coords = np.linspace(-axis_limit, axis_limit, 100)
                    ax.plot(
                        line_coords,
                        line_coords,
                        color="gray",
                        linestyle="--",
                        label="decision boundary",
                    )
                    ax.legend()

        plt.tight_layout(
            rect=[0, 0, 1, 0.96]
        )  # Adjust layout to make room for suptitle

    # Create and run the animation
    ani = FuncAnimation(fig, update, frames=hp.epochs, repeat=False)
    ani.save(filename="anim.gif", writer="imagemagick")


if __name__ == "__main__":
    main()
