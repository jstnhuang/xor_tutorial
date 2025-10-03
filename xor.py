from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class HyperParameters:
    lr: float = 1
    epochs: int = 25


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

    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    hp = HyperParameters()
    model = XorNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=hp.lr)

    for epoch in range(hp.epochs):
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            correct = (pred.argmax(1) == y).sum().item()

        print(f"Epoch {epoch+1}/{hp.epochs}, loss: {loss:>4f}, acc: {correct/4.0}")
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")
    print(model.state_dict())
    linear1 = model.state_dict()["fc1.weight"].detach()
    bias1 = model.state_dict()["fc1.bias"].detach()
    linear2 = model.state_dict()["fc2.weight"].detach()
    bias2 = model.state_dict()["fc2.bias"].detach()
    axis_limit = 1.75
    with torch.no_grad():
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for X, y in data_loader:
            tf1 = model.fc1(X)
            relu = F.relu(tf1)
            tf2 = model.fc2(relu)

            for i, res in enumerate([tf1, relu, tf2]):
                ax = axes[i]
                ax.scatter(res[y == 0, 0], res[y == 0, 1], c="blue", label="0")
                ax.scatter(res[y == 1, 0], res[y == 1, 1], c="red", label="1")
                ax.set_title(f"layer {i}")
                ax.set_xlim(-axis_limit, axis_limit)
                ax.set_ylim(-axis_limit, axis_limit)
                ax.legend()
                ax.grid(True)
                ax.set_aspect("equal", adjustable="box")
                ax.axhline(0)
                ax.axvline(0)
                if i == 2:
                    line_coords = np.linspace(-axis_limit, axis_limit, 100)
                    ax.plot(
                        line_coords,
                        line_coords,
                        color="gray",
                        label="decision boundary",
                    )
                    ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
