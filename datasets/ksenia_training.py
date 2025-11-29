import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os


##########################################
# Dataset
##########################################

class KseniaDataset(Dataset):
    def __init__(self, csv_path, user="ksenia", scaler=None):
        df = pd.read_csv(csv_path)

        # columns = subject, sessionIndex, rep, feat1, feat2, ...
        if "rep" in df.columns:
            feat_start = list(df.columns).index("rep") + 1
        else:
            feat_start = 3

        feature_cols = df.columns[feat_start:]
        df["subject"] = df["subject"].astype(str)

        # filter only Ksenia
        df = df[df["subject"] == user]

        X = df[feature_cols].astype(float).values

        # scale features
        if scaler is None:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            X = self.scaler.transform(X)

        self.X = X.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32)



##########################################
# One-Class Embedding Network
##########################################

class KseniaNet(nn.Module):
    def __init__(self, input_dim, emb_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim)
        )

    def forward(self, x):
        emb = self.model(x)
        emb = nn.functional.normalize(emb, p=2, dim=-1)
        return emb



##########################################
# TRAINING
##########################################

def train_ksenia_model(csv_path,
                       out_dir="ksenia_auth_model",
                       batch=32, epochs=20, lr=1e-3):

    os.makedirs(out_dir, exist_ok=True)

    # Load dataset (only Ksenia)
    temp_ds = KseniaDataset(csv_path)
    input_dim = temp_ds.X.shape[1]
    scaler = temp_ds.scaler
    joblib.dump(scaler, f"{out_dir}/scaler.pkl")

    dataset = KseniaDataset(csv_path, scaler=scaler)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = KseniaNet(input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss ensures embeddings cluster around their mean
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for X in loader:
            X = X.to(device)
            emb = model(X)

            # center = mean embedding of batch
            center = emb.mean(dim=0, keepdim=True)

            # loss = distance to centroid (one-class objective)
            loss = ((emb - center) ** 2).sum(dim=1).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        print(f"Epoch {epoch}: loss={np.mean(losses):.6f}")

    # Compute final centroid and variance for threshold
    model.eval()
    all_embs = []
    with torch.no_grad():
        for X in loader:
            X = X.to(device)
            all_embs.append(model(X).cpu().numpy())

    all_embs = np.vstack(all_embs)
    mu = all_embs.mean(axis=0)
    sigma = all_embs.std(axis=0)

    # define threshold = avg L2 distance + 3 std
    dists = np.linalg.norm(all_embs - mu, axis=1)
    threshold = 0.1

    np.save(f"{out_dir}/mu.npy", mu)
    np.save(f"{out_dir}/sigma.npy", sigma)
    np.save(f"{out_dir}/threshold.npy", threshold)
    torch.save(model.state_dict(), f"{out_dir}/model.pt")

    print("\nMODEL TRAINED")
    print("Centroid + threshold saved.")
    print("Threshold =", threshold)



##########################################
# VERIFY FUNCTION
##########################################

def verify(sample_features, model_dir="ksenia_auth_model"):
    """Return True if sample is from Ksenia."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = joblib.load(f"{model_dir}/scaler.pkl")
    mu = np.load(f"{model_dir}/mu.npy")
    threshold = np.load(f"{model_dir}/threshold.npy")

    model = KseniaNet(len(sample_features))
    model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location=device))
    model.to(device)
    model.eval()

    x = scaler.transform([sample_features])
    x = torch.tensor(x, dtype=torch.float32).to(device)

    with torch.no_grad():
        emb = model(x).cpu().numpy()[0]

    dist = np.linalg.norm(emb - mu)
    return dist < threshold, dist



##########################################
# Example Usage
##########################################

if __name__ == "__main__":
    # TRAIN
    train_ksenia_model("ksenia_training_2.csv")

    # TEST / VERIFY
    # Suppose this is a SINGLE keystroke feature row
    sample = [0.07275581359863281,0.17601799964904785,0.10326218605041504,0.08356308937072754,0.12697720527648926,0.04341411590576172,0.049411773681640625,0.08414387702941895,0.03473210334777832,0.09474897384643555,0.2240900993347168,0.12934112548828125,0.09111189842224121,0.3291647434234619,0.2380528450012207,0.12380099296569824,0.2374434471130371,0.11364245414733887,0.08602690696716309,0.13656139373779297,0.05053448677062988,0.08996963500976562,0.15062522888183594,0.06065559387207031,0.05662393569946289,0.2348320484161377,0.1782081127166748,0.08917522430419922
]   # <--- put actual features

    is_ksenia, score = verify(sample)
    print("Authenticated:", is_ksenia, "distance:", score)

    sample = [0.1491,0.3979,0.2488,0.1069,0.1674,0.0605,0.1169,0.2212,0.1043,0.1417,1.1885,1.0468,0.1146,1.6055,1.4909,0.1067,0.759,0.6523,0.1016,0.2136,0.112,0.1349,0.1484,0.0135,0.0932,0.3515,0.2583,0.1338
]  # <--- put actual features

    is_ksenia, score = verify(sample)
    print("Authenticated:", is_ksenia, "distance:", score)

    sample = [0.08061408996582031,0.2100238800048828,0.1294097900390625,0.0618894100189209,0.09592700004577637,0.03403759002685547,0.05246281623840332,0.10519766807556152,0.0527348518371582,0.08765697479248047,0.2594735622406006,0.17181658744812012,0.07918453216552734,0.3579275608062744,0.27874302864074707,0.1203317642211914,0.2019655704498291,0.0816338062286377,0.06158113479614258,0.06273794174194336,0.0011568069458007812,0.09793233871459961,0.12535810470581055,0.027425765991210938,0.04299592971801758,0.10731291770935059,0.06431698799133301,0.0816793441772461
]  # <--- put actual features

    is_ksenia, score = verify(sample)
    print("Authenticated:", is_ksenia, "distance:", score)
