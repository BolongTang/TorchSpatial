# Run this under the directory that contains TorchSpatial, not under TorchSpatial itself

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from sklearn.model_selection import train_test_split

from TorchSpatial.modules.trainer import train, forward_with_np_array
from TorchSpatial.modules.encoder_selector import get_loc_encoder
from TorchSpatial.modules.model import ThreeLayerMLP

from pathlib import Path
import numpy as np

# Using birdsnap
def main():
    device = "cpu"
    num_classes = 500
    img_dim = loc_dim = embed_dim = 784 # Assumed, can change.
    coord_dim = 2

    # - fake dataset: each row = (img_emb[784], latlon[2], class_index)
    N = 2048
    img = torch.randn(N, img_dim) * 10                         # [N,784]

    lat = torch.rand(N)*180 - 90 # - 90 to 90
    lon = torch.rand(N)*360 # 0 to 360
    loc = torch.stack([lat, lon], dim=1) 

    y = torch.randint(0, num_classes, (N,), dtype=torch.long)

    # Structured relationship: class depends on latitude and longitude bands
    num_lat_bands = 10
    num_lon_bands = 10
    lat_band = ((lat + 90) // (180 / num_lat_bands)).long()
    lon_band = (lon // (360 / num_lon_bands)).long()
    y = (lat_band * num_lon_bands + lon_band) % num_classes

    Ximg_tr, Ximg_te, Xloc_tr, Xloc_te, y_tr, y_te = train_test_split(
    img, loc, y, test_size=0.2, random_state=42, shuffle=True
    )

    train_data = list(zip(Ximg_tr, Xloc_tr, y_tr))
    test_data  = list(zip(Ximg_te, Xloc_te, y_te))

    # - Dataloader (loads image embeddings)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)

    # - location encoder
    # Allowed: Space2Vec-grid, Space2Vec-theory, xyz, NeRF, Sphere2Vec-sphereC, Sphere2Vec-sphereC+, Sphere2Vec-sphereM, Sphere2Vec-sphereM+, Sphere2Vec-dfs, rbf, rff, wrap, wrap_ffn, tile_ffn, Siren(SH)
    # overrides is a dictionary that allows overriding specific params. 
    # ex. loc_encoder = get_loc_encoder(name = "Space2Vec-grid", overrides = {"max_radius":7800, "min_radius":15, "spa_embed_dim":784})
    # For other required arguments, please refer to the docs (ex. rbf)
    # https://torchspatial.readthedocs.io/en/latest/2D%20Location%20Encoders/rbf.html
    loc_encoder_name = "Siren(SH)"
    loc_encoder = get_loc_encoder(name = loc_encoder_name, overrides = {"spa_embed_dim":784, "device": device}) # "device": device is needed if you defined device = 'cpu' above and don't have cuda setup to prevent "AssertionError: Torch not compiled with CUDA enabled", because the default is device="cuda"
    

    # - model
    # decoder = ThreeLayerMLP(input_dim = 784, hidden_dim = 1024, category_count = 512)
    decoder = ThreeLayerMLP(input_dim = embed_dim, hidden_dim = 1024, category_count = num_classes).to(device)
    
    # - Criterion
    criterion = nn.CrossEntropyLoss()
    # - Optimizer
    optimizer = Adam(params = list(loc_encoder.ffn.parameters()) + list(decoder.parameters()), lr = 1e-3)
    # - train() 
    epochs = 15
    train(epochs = epochs, 
            batch_count_print_avg_loss = 30,
            loc_encoder = loc_encoder,
            dataloader = train_loader,
            decoder = decoder,
            criterion = criterion,
            optimizer = optimizer,
            device = device)
    
    # - test
    loc_encoder.eval()
    decoder.eval()

    total = 0
    correct_top1 = 0
    correct_top3 = 0
    mrr_sum = 0

    with torch.no_grad():
        for img_b, loc_b, y_b in test_loader:
            img_b, loc_b, y_b = img_b.to(device), loc_b.to(device), y_b.to(device)

            img_embedding = img_b
            loc_embedding = forward_with_np_array(batch_data=loc_b, model=loc_encoder)

            loc_img_interaction_embedding = torch.mul(loc_embedding, img_embedding)
            logits = decoder(loc_img_interaction_embedding)

            # Top-1
            pred = logits.argmax(dim=1)

            # Top-3 accuracy
            top3_idx = logits.topk(3, dim=1).indices                    # [B, 3]
            correct_top3 += (top3_idx == y_b.unsqueeze(1)).any(dim=1).sum().item()

            # MRR (full ranking over all classes)
            ranking = logits.argsort(dim=1, descending=True)             # [B, C]
            positions = ranking.argsort(dim=1)                           # [B, C] where positions[b, c] = rank index (0-based)
            true_pos0 = positions.gather(1, y_b.view(-1, 1)).squeeze(1)  # [B]
            mrr_sum += (1.0 / (true_pos0.float() + 1.0)).sum().item()

            total += y_b.size(0)
            correct_top1 += (pred == y_b).sum().item()

    top1_acc = 100.0 * correct_top1 / total if total else 0.0
    top3_acc = 100.0 * correct_top3 / total if total else 0.0
    mrr = mrr_sum / total if total else 0.0

    print(f"Top-1 Accuracy on {total} test images: {top1_acc:.2f}%")
    print(f"Top-3 Accuracy on {total} test images: {top3_acc:.2f}%")
    print(f"MRR on {total} test images: {mrr:.4f}")

    # - save model
    model_name = f"{loc_encoder_name}.pt"
    path = Path(f"TorchSpatial/checkpoints/{model_name}")
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "epoch": epochs,
        "loc_encoder": loc_encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)

    print(f"Model saved as TorchSpatial/checkpoints/{model_name}")


if __name__ == "__main__":
    main()