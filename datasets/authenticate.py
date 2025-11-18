from test import TripletSNN, CMUDatasetTriplet
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset and model
dataset = CMUDatasetTriplet("ksenia_test_data.csv")
input_dim = dataset.X.shape[1]
print(f"input dim: {input_dim}")

model = TripletSNN(input_dim=input_dim)
ckpt = torch.load("../models/snn_final.pt", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

# ---- Embed sample helper ---
def embed_sample(sample_vector, model):
    model.eval()
    x = torch.tensor(sample_vector).float().unsqueeze(0).unsqueeze(0)  # (1,1,F)
    lengths = torch.tensor([1])
    with torch.no_grad():
        emb = model.subnet(x.to(device), lengths.to(device))
    return emb.cpu().numpy()[0]

# ---- Take the first two rows ----
sample1 = dataset.X[1]    # row 1 (reference)
sample2 = dataset.X[0]    # row 2 (probe)

print("User 1:",sample1)
print("User 2:",sample2)

user1_label = dataset.y[1]
user2_label = dataset.y[0]

print("Row 1 user:", user1_label)
print("Row 2 user:", user2_label)

# ---- Embed both ----
emb1 = embed_sample(sample1, model)
emb2 = embed_sample(sample2, model)

# ---- Distance between them ----
dist = np.linalg.norm(emb1 - emb2)
print("Embedding distance:", dist)

# ---- Choose a threshold ----
# If you already computed your optimal threshold earlier, use it here
# Otherwise a typical SNN threshold is around 0.9–1.2 for L2-normalized vectors
threshold = 0.4

same_user = dist < threshold

print("\n--- RESULT ---")
print("Same user?" , same_user)
print("Distance =", dist)
print("Threshold =", threshold)
