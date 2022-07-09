import copy

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data

from graph_ml.model import GCN
from graph_ml.ogb_wrapper import Evaluator, PygNodePropPredDataset
from graph_ml.trainer import evaluate, train_a_step

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = "ogbn-arxiv"
    dataset = PygNodePropPredDataset(
        name=dataset_name, root="dataset", transform=T.ToSparseTensor()
    )
    data: Data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"].to(device)

    args = {
        "device": device,
        "num_layers": 4,
        "hidden_dim": 256,
        "dropout": 0.5,
        "lr": 0.01,
        "epochs": 200,
    }

    model = GCN(
        data.num_features,
        args["hidden_dim"],
        dataset.num_classes,
        args["num_layers"],
        args["dropout"],
    ).to(device)
    model.reset_parameters()

    evaluator = Evaluator(name="ogbn-arxiv")

    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
    loss_fn = F.nll_loss

    best_model = None
    best_valid_acc = 0

    for epoch in range(1, 1 + args["epochs"]):
        loss = train_a_step(model, data, train_idx, optimizer, loss_fn)
        result = evaluate(model, data, split_idx, evaluator)
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

    best_result = evaluate(
        best_model, data, split_idx, evaluator, save_model_results=True
    )
    train_acc, valid_acc, test_acc = best_result
    print(
        f"Best model: "
        f"Train: {100 * train_acc:.2f}%, "
        f"Valid: {100 * valid_acc:.2f}% "
        f"Test: {100 * test_acc:.2f}%"
    )
