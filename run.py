import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torchsummary import summary
from tqdm import tqdm


from convmixer import convmixer
from generate_plot import draw
from mnist import Mnist
from CIFAR10 import Cifar10
from resnet import resnet


def print_outputs(outputs: torch.Tensor, labels: torch.Tensor):
    """prints the outputs and labels in a pretty table

    Args:
        outputs (torch.Tensor): printed in the prediction cols
        labels (torch.Tensor): printed in the labels cols
    """
    ang_lbl = torch.rad2deg(torch.atan2(labels[:, 0], labels[:, 1]))
    ang_pred = torch.rad2deg(torch.atan2(outputs[:, 0], outputs[:, 1]))

    ang_pred = ang_pred.data.cpu().numpy()
    ang_lbl = ang_lbl.data.cpu().numpy()
    outputs = outputs.data.cpu().numpy()
    labels = labels.data.cpu().numpy()

    print(f"pred sin | true sin | pred cos | true cos | pred ang | true ang")
    print("-" * 41)
    for i in range(len(labels)):
        print(
            f"{outputs[i][0]:8.4} | {labels[i][0]:8.4} | {outputs[i][1]:8.4} | {labels[i][1]:8.4} | {ang_pred[i]:8.4} | {ang_lbl[i]:8.4}"
        )


def abs_angle_error_pytorch(
    y_pred: torch.FloatTensor, y_true: torch.FloatTensor
) -> torch.FloatTensor:
    """angle difference given 2 angles in radians? between 0...1

    Args:
        y_true (float): actual angle 
        y_pred (float): predicted angle
    """
    diff = torch.abs(y_pred - y_true)
    return torch.min(360 - diff, diff)


def squared_angle_error_pytorch(
    y_true: torch.FloatTensor, y_pred: torch.FloatTensor
) -> torch.FloatTensor:
    """squared angle difference given 2 angles in radians? between 0...1

    Args:
        y_true (float): actual angle 
        y_pred (float): predicted angle
    """
    diff = y_pred - y_true
    diff = torch.min(360 - diff, diff)
    return diff ** 2


def run_convmixer(
    sparsity: float,
    num_epochs: int = 10,
    batch_size: int = 128,
    print_summary: bool = False,
):
    """trains the convmixer model to estimate mnist image rotation

    Args:
        sparsity (float): sparsity for convmixer model
        num_epochs (int, optional): number of epochs to train. Defaults to 10.
        batch_size (int, optional): batch size for training. Defaults to 128.
        print_summary (bool, optional): wether to print torchsummary of the 
        model. Defaults to False.

    Returns:
        acc (int): test accuracy of final epoch
    """
    ds = Cifar10(train=True)
    val_ds = Cifar10(train=False)
    device = torch.device("cuda")
    batch_size = 128
    num_epochs = 7
    sparsity = min(sparsity, 1.0)

    train_loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4
    )

    dl = {"train": train_loader, "val": val_loader}

    model = convmixer(
        dim=1024,
        depth=8,
        n_classes=2,
        sparsity=sparsity,
        input_channels=ds.input_channels,
    )
    model = model.to(device)
    if print_summary:
        summary(model, (ds.input_channels, 28, 28))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [3, 7], gamma=0.5)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch} - sparsity {sparsity}")
        print("-" * 40)

        for phase in ["train", "val"]:
            t0 = time.time()
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            tot = len(dl[phase])

            for batch, (inputs, labels) in tqdm(enumerate(dl[phase]), total=tot):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    preds = torch.atan2(outputs[:, 0], outputs[:, 1])
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(
                    torch.abs(preds - torch.atan2(labels[:, 0], labels[:, 1]))
                    <= 0.349  # 20 deg ~ 0.34906 radiants
                )
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / (len(dl[phase]) * batch_size)
            epoch_acc = running_corrects.double() / (len(dl[phase]) * batch_size)
            t1 = time.time()

            print(
                f"{phase} Loss: {epoch_loss:.4} Acc: {epoch_acc:.5} in {t1-t0:.4} seconds"
            )
        print("=" * 40)

    print_outputs(outputs, labels)

    return epoch_acc


if __name__ == "__main__":
    results = []
    times = []
    s = [i / 100.0 for i in range(1, 101,5)]
    for sparsity in [0.001]+s:
        t0 = time.time()
        acc = run_convmixer(sparsity)
        t1 = time.time()
        results.append(acc)
        times.append(t1 - t0)

    with open("sparsity_out.csv", "w", newline="") as file:
        writer = csv.writer(file)

        print("----------+---------------------")
        print(" Sparsity | Final Test Accuracy")
        print("----------+---------------------")
        for sparsity, acc, t in zip(s, results, times):
            print(f" {sparsity:8.5} | {acc:7.5}")
            writer.writerow([sparsity, f"{acc:.7}", f"{t:.4}"])
    draw()
