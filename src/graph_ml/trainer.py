import pandas as pd
import torch
from ogb.nodeproppred import Evaluator

from graph_ml.model import GCN


def train(model: GCN, data, train_idx, optimizer, loss_fn):
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
        df.to_csv("local_ogbn-arxiv_node.csv", sep=",", index=False)

    return train_acc, valid_acc, test_acc
