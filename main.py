# Run this under the directory that contains TorchSpatial, not under TorchSpatial itself

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from sklearn.model_selection import train_test_split

from TorchSpatial.modules.trainer import train, forward_with_np_array
from TorchSpatial.modules.encoder_selector import get_loc_encoder
from TorchSpatial.modules.model import ThreeLayerMLP
import TorchSpatial.utils.datasets as data_import

from pathlib import Path
import numpy as np


def main():

    task = "Regression" # "Classification" or "Regression"

    # - import dataset
    params = {"dataset": "birdsnap", "meta_type": "orig_meta", "regress_dataset": []}
    eval_split = "train"
    train_remove_invalid = True
    eval_remove_invalid = True

    all_data = data_import.load_dataset(params = params,
        eval_split = eval_split,
        train_remove_invalid = train_remove_invalid,
        eval_remove_invalid = eval_remove_invalid,
        load_cnn_predictions=True,
        load_cnn_features=True,
        load_cnn_features_train=True)

    # Using birdsnap
    # - birdsnap dataset
    dataset = "birdsnap"
    task = "Classification"
    N = 19133
    device = "cpu"
    num_classes = 500 # birdsnap class count
    img_dim = loc_dim = embed_dim = 2048 # birdsnap embedding count
    coord_dim = 2 #lonlat

    img = all_data["val_feats"] # shape=(19133, 2048)
    loc = all_data["val_locs"] # shape=(19133, 2)
    y = all_data["val_preds"] # shape=(19133, 500)

    if task == "Classification":
        embed_dim = img_dim 
    elif task == "Regression": 
        embed_dim = img_dim + loc_dim

    # ------------------------
    # Train / Test split
    # ------------------------
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
    loc_encoder_name = "Space2Vec-grid"
    loc_encoder = get_loc_encoder(name = loc_encoder_name, overrides = {"spa_embed_dim": loc_dim, "coord_dim":coord_dim, "device":device}).to(device) # "device": device is needed if you defined device = 'cpu' above and don't have cuda setup to prevent "AssertionError: Torch not compiled with CUDA enabled", because the default is device="cuda"

    # - model
    # - Criterion
    if task == "Classification":
        decoder = ThreeLayerMLP(input_dim = embed_dim, hidden_dim = 1024, category_count = num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
    elif task == "Regression":
        decoder = ThreeLayerMLP(input_dim = embed_dim, hidden_dim = 1024, category_count = y.size()[1]).to(device)
        criterion = nn.MSELoss()

    # - Optimizer
    optimizer = Adam(params = list(loc_encoder.parameters()) + list(decoder.parameters()), lr = 1e-3)
    # - train() 
    epochs = 10
    train(task = task,
            epochs = epochs, 
            batch_count_print_avg_loss = 10,
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

            if task == "Classification":
                loc_img_interaction_embedding = torch.mul(loc_embedding, img_embedding)
                logits = decoder(loc_img_interaction_embedding)

                # Top-1
                pred = logits.argmax(dim=1)
                y_b = y_b.argmax(dim=1) # dataset has proba lists, here converting to class indices

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

            elif task == "Regression":
                loc_img_concat_embedding = torch.cat((loc_embedding, img_embedding), dim = 1)
                yhat = decoder(loc_img_concat_embedding)

                # r-square
                # Compute per-sample mean over feature dimension
                y_mean = torch.mean(y_b, dim=1, keepdim=True)          # (B, 1)

                ss_res = torch.sum((y_b - yhat) ** 2, dim=1)           # (B,)
                ss_tot = torch.sum((y_b - y_mean) ** 2, dim=1)         # (B,)

                r2 = 1 - ss_res / ss_tot                               # (B,)
                r2 = torch.mean(r2)                                    # scalar

                # MAE
                mae = torch.mean(torch.abs(yhat - y_b))

                # RMSE
                rmse = torch.sqrt(torch.mean((yhat - y_b) ** 2))

                total += y_b.size(0)
            

    if task == "Classification":
        top1_acc = 100.0 * correct_top1 / total if total else 0.0
        top3_acc = 100.0 * correct_top3 / total if total else 0.0
        mrr = mrr_sum / total if total else 0.0

        print(f"Top-1 Accuracy on {total} test images: {top1_acc:.2f}%")
        print(f"Top-3 Accuracy on {total} test images: {top3_acc:.2f}%")
        print(f"MRR on {total} test images: {mrr:.4f}")

    elif task == "Regression":
        print(f"r-square on {total} test images: {r2:.2f}%")
        print(f"MAE of pred on {total} test images: {mae:.2f}%")
        print(f"RMSE of pred on {total} test images: {rmse:.2f}%")

    # - save model
    model_path = f"TorchSpatial/checkpoints/{loc_encoder_name}_{epochs}_{task[:3]}.pt"
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "epoch": epochs,
        "loc_encoder": loc_encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)

    print(f"Model saved as {model_path}")


if __name__ == "__main__":
    main()