import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

from sklearn.model_selection import train_test_split

from modules.trainer import train
from modules.encoder_selector import get_loc_encoder
from modules.model import ThreeLayerMLP

# Using birdsnap
def main():
    device = "cpu"
    num_classes = 500
    img_dim = loc_dim = embed_dim = 784 # Assumed, can change.
    coord_dim = 2

    # - fake dataset: each row = (img_emb[784], latlon[2], class_index)
    N = 2048
    img = torch.randn(N, img_dim)                          # [N,784]
    loc = torch.randn(N, coord_dim)                        # [N,2]
    y = torch.randint(0, num_classes, (N,), dtype=torch.long)

    Ximg_tr, Ximg_te, Xloc_tr, Xloc_te, y_tr, y_te = train_test_split(
    img, loc, y, test_size=0.2, random_state=42, stratify=y
    )

    # - Dataloader (loads image embeddings)
    train_loader = DataLoader(
    TensorDataset(Ximg_tr, Xloc_tr, y_tr),
    batch_size=32,
    shuffle=True
    )

    test_loader = DataLoader(
    TensorDataset(Ximg_te, Xloc_te, y_te),
    batch_size=32,
    shuffle=False
    )

    # - location encoder
    # Allowed: Space2Vec-grid, Space2Vec-theory, xyz, NeRF, Sphere2Vec-sphereC, Sphere2Vec-sphereC+, Sphere2Vec-sphereM, Sphere2Vec-sphereM+, Sphere2Vec-dfs, rbf, rff, wrap, wrap_ffn, tile_ffn
    # overrides is a dictionary that allows overriding specific params. 
    # ex. loc_encoder = get_loc_encoder(name = "Space2Vec-grid", overrides = {"max_radius":7800, "min_radius":15, "spa_embed_dim":784})
    loc_encoder = get_loc_encoder(name = "Space2Vec-grid", overrides = {"coord_dim": coord_dim, "spa_embed_dim": loc_dim}).to(device)

    # - model
    # decoder = ThreeLayerMLP(input_dim = 784, hidden_dim = 1024, category_count = 512)
    decoder = ThreeLayerMLP(input_dim = embed_dim, hidden_dim = 1024, category_count = num_classes)
    model = nn.Sequential(loc_encoder, decoder).to(device)
    
    # - Criterion
    criterion = nn.CrossEntropyLoss()
    # - Optimizer
    optimizer = Adam(params = model.parameters(), lr = 1e-3)
    # - train() 
    train(epochs = 30, 
            batch_count_print_avg_loss = 2000,
            loc_encoder = loc_encoder,
            dataloader = train_loader,
            model = model,
            criterion = criterion,
            optimizer = optimizer,
            device = device)
    
    # - test
    model.eval()
    total = correct = 0
    with torch.no_grad():
        for img_b, loc_b, y_b in test_loader:
            img_b, loc_b, y_b = img_b.to(device), loc_b.to(device), y_b.to(device)
            out = model(img_b, loc_b)
            pred = out.argmax(dim=1)
            total += y_b.size(0)
            correct += (pred == y_b).sum().item()

    print(f"Accuracy of the network on the {total} test images: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()