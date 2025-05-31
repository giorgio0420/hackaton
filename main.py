import argparse
import os
import torch
import numpy as np
import random
from source.loadData import GraphDataset
from source.models import GNN
from source.transform import add_node_degree_feature, add_clustering_coefficient, normalize_edge_attr
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
import logging
import pandas as pd

# Seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def get_folder_name(path):
    return os.path.basename(os.path.dirname(path))

class NoisyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, p_noisy):
        super().__init__()
        self.p = p_noisy
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        targets = targets.to(logits.device)  # <--- Sicurezza extra
        losses = self.ce(logits, targets)
        weights = (1 - self.p) + self.p * (1 - torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float().sum(dim=1))
        weights = weights.to(logits.device)
        return (losses * weights).mean()


model = GNN(
    num_class=6,
    num_layer=2,
    emb_dim=128,
    input_dim=input_dim,
    residual=False,
    drop_ratio=0.2,
    JK='last',
    graph_pooling='mean'
)


def train_model(train_loader, model, optimizer, criterion, device, folder_name):
    logs_folder = os.path.join("logs")
    os.makedirs(logs_folder, exist_ok=True)
    log_file = os.path.join(logs_folder, f"training_{folder_name}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    best_val_acc = 0
    for epoch in range(1, 51):  # 50 epoche
        model.train()
        total_loss, correct, total = 0, 0, 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data.y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == data.y.to(device)).sum().item()
            total += data.y.size(0)
        acc = correct / total
        logging.info(f"Epoch {epoch} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {acc:.4f}")

        if epoch % 10 == 0:
            logging.info(f"--- Checkpoint at epoch {epoch} ---")
        if epoch % 10 == 0 or acc > best_val_acc:
            checkpoint_path = f"checkpoints/model_{folder_name}_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")
            best_val_acc = acc

def predict(test_loader, model, device, folder_name):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
    os.makedirs("submission", exist_ok=True)
    df = pd.DataFrame({"id": list(range(len(predictions))), "pred": predictions})
    df.to_csv(f"submission/testset_{folder_name}.csv", index=False)
    print(f"Predictions saved: submission/testset_{folder_name}.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, required=True, help="Path to test.json.gz")
    parser.add_argument("--train_path", type=str, help="Path to train.json.gz")
    args = parser.parse_args()

    folder_name = get_folder_name(args.test_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carica dati
    test_dataset = GraphDataset(args.test_path)
    transformed_test = []
    for data in test_dataset:
        #data = normalize_edge_attr(data)
        data = add_node_degree_feature(data)
        data = add_clustering_coefficient(data)
        transformed_test.append(data)
    test_loader = DataLoader(transformed_test, batch_size=32, shuffle=False)

    # Modello
    input_dim = transformed_test[0].x.shape[1]
    model = GNN(num_class=6, input_dim=input_dim).to(device)

    if args.train_path:
        train_dataset = GraphDataset(args.train_path)
        transformed_train = []
        for data in train_dataset:
            #data = normalize_edge_attr(data)
            data = add_node_degree_feature(data)
            data = add_clustering_coefficient(data)
            transformed_train.append(data)
        train_loader = DataLoader(transformed_train, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        train_model(train_loader, model, optimizer, criterion, device, folder_name)
    else:
        checkpoint_path = f"checkpoints/model_{folder_name}_best.pth"  # Specifica il checkpoint finale corretto
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model from {checkpoint_path}")

    predict(test_loader, model, device, folder_name)

if __name__ == "__main__":
    main()
