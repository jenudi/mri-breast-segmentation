import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from src.preprocessing import MRIDataset
from src.model import UNet
from src.eval import dice_loss

def train_unet(
    train_df,
    val_df,
    batch_size,
    epochs=40,
    patience=5,
    learning_rate=1e-4,
    SEED=42,
):
    # Create datasets
    train_dataset = MRIDataset(train_df, is_train=True)
    val_dataset   = MRIDataset(val_df,   is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Initialize U-Net
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = UNet(n_channels=1, n_classes=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-5)

    best_val_loss = float("inf")
    best_model_state = None
    counter = 0

    epoch_loss_trn = []
    epoch_loss_val = []

    print("Starting U-Net training...")

    #  TRAIN LOOP
    for epoch in range(epochs):

        # ---- TRAIN ----
        model.train()
        train_losses = []
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

        for x, y, pid, slice_num in progress_bar:
            x = x.to(device)   # (B,1,512,512)
            y = y.to(device)   # (B,1,512,512)

            logits = model(x)
            loss = dice_loss(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            progress_bar.set_postfix(train_loss=np.mean(train_losses))

        epoch_loss_trn.append(np.mean(train_losses))

        # ---- VALIDATION ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Validating Epoch {epoch+1}")
            for x, y, pid, slice_num in progress_bar:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = dice_loss(logits, y)
                val_losses.append(loss.item())

                progress_bar.set_postfix(val_loss=np.mean(val_losses))

        current_val_loss = np.mean(val_losses)
        epoch_loss_val.append(current_val_loss)

        # EARLY STOPPING LOGIC
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter}/{patience}")

        if counter >= patience:
            print("Early stopping triggered!")
            break

    #  PLOT LOSSES EACH EPOCH
    plt.figure(figsize=(10,5))
    plt.plot(epoch_loss_trn, label="Training Loss")
    plt.plot(epoch_loss_val, label="Validation Loss")
    plt.title("Dice Loss During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Save best model
    #final_path = f"{model_out_dir}/unet_best.pth"
    #torch.save(best_model_state, final_path)
    #print("Model saved to:", final_path)

    return model
