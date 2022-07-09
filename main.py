import copy

import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import Evaluator, dataset_pyg

from graph_ml.model import GCN

dataset_name = "ogbn-arxiv"
# Load the dataset and transform it to sparse tensor
dataset = dataset_pyg.PygNodePropPredDataset(
    name=dataset_name, root=".", transform=T.ToSparseTensor()
)
print("The {} dataset has {} graph".format(dataset_name, len(dataset)))

# Extract the graph
data = dataset[0]
print(data)


# Make the adjacency matrix to symmetric
data.adj_t = data.adj_t.to_symmetric()

device = "cuda" if torch.cuda.is_available() else "cpu"

# If you use GPU, the device should be cuda
print("Device: {}".format(device))

data = data.to(device)
split_idx = dataset.get_idx_split()
train_idx = split_idx["train"].to(device)


def train(model, data, train_idx, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.adj_t)
    loss = loss_fn(out[train_idx], data.y.squeeze(1)[train_idx])

    loss.backward()
    optimizer.step()

    return loss.item()


# Test function here
@torch.no_grad()
def test(model, data, split_idx, evaluator: Evaluator, save_model_results=False):

    model.eval()
    out = model(data.x, data.adj_t)

    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval(
        {
            "y_true": data.y[split_idx["train"]],
            "y_pred": y_pred[split_idx["train"]],
        }
    )["acc"]
    valid_acc = evaluator.eval(
        {
            "y_true": data.y[split_idx["valid"]],
            "y_pred": y_pred[split_idx["valid"]],
        }
    )["acc"]
    test_acc = evaluator.eval(
        {
            "y_true": data.y[split_idx["test"]],
            "y_pred": y_pred[split_idx["test"]],
        }
    )["acc"]

    if save_model_results:
        print("Saving Model Predictions")

        data = {}
        data["y_pred"] = y_pred.view(-1).cpu().detach().numpy()

        df = pd.DataFrame(data=data)
        # Save locally as csv
        df.to_csv("ogbn-arxiv_node.csv", sep=",", index=False)

    return train_acc, valid_acc, test_acc


args = {
    "device": device,
    "num_layers": 3,
    "hidden_dim": 256,
    "dropout": 0.5,
    "lr": 0.01,
    "epochs": 100,
}
args

model = GCN(
    data.num_features,
    args["hidden_dim"],
    dataset.num_classes,
    args["num_layers"],
    args["dropout"],
).to(device)
evaluator = Evaluator(name="ogbn-arxiv")


# reset the parameters to initial random value
model.reset_parameters()

optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
loss_fn = F.nll_loss

best_model = None
best_valid_acc = 0

for epoch in range(1, 1 + args["epochs"]):
    loss = train(model, data, train_idx, optimizer, loss_fn)
    result = test(model, data, split_idx, evaluator)
    train_acc, valid_acc, test_acc = result
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_model = copy.deepcopy(model)
    print(
        f"Epoch: {epoch:02d}, "
        f"Loss: {loss:.4f}, "
        f"Train: {100 * train_acc:.2f}%, "
        f"Valid: {100 * valid_acc:.2f}% "
        f"Test: {100 * test_acc:.2f}%"
    )


best_result = test(best_model, data, split_idx, evaluator, save_model_results=True)
train_acc, valid_acc, test_acc = best_result
print(
    f"Best model: "
    f"Train: {100 * train_acc:.2f}%, "
    f"Valid: {100 * valid_acc:.2f}% "
    f"Test: {100 * test_acc:.2f}%"
)
