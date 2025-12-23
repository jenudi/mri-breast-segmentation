import torch
from typing import Iterable, Tuple
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch.nn.functional as F
import pandas as pd

from src.preprocessing import MRITestDataset


def dice_loss(pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    pred   : raw logits from UNet, shape (B,1,H,W)
    target : ground truth mask, shape (B,1,H,W), values 0 or 1
    """
    pred = torch.sigmoid(pred)      # convert logits → probability
    pred_flat = pred.contiguous().view(pred.size(0), -1)
    target_flat = target.contiguous().view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()   # minimize 1 - Dice


def dice_score_binary(
    pred_bin: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6
) -> torch.Tensor:
    """
    Compute Dice score for binary segmentation masks.

    Args:
        pred_bin: Binary prediction mask of shape (B, 1, H, W), values in {0,1}.
        target: Ground truth binary mask of shape (B, 1, H, W).
        smooth: Small constant to avoid division by zero.

    Returns:
        Mean Dice score over the batch as a scalar tensor.
    """
    pred_flat = pred_bin.contiguous().view(pred_bin.size(0), -1)
    target_flat = target.contiguous().view(target.size(0), -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean()


def find_best_threshold(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate a trained segmentation model over multiple probability thresholds
    and identify the threshold that yields the highest Dice score.

    This is necessary because the U-Net outputs *probabilities* (after sigmoid),
    but segmentation requires *binary masks*. The point at which we binarize
    (0.3? 0.5? 0.9?) has a strong impact on the Dice score.

    Workflow:
    ---------
    1. Generate a list of thresholds between 0.05 and 0.95.
    2. For each threshold:
         - Run the model on the full validation set.
         - Convert probabilities → binary masks using the threshold.
         - Compute Dice (1 - dice_loss) for each batch.
         - Store the mean Dice for that threshold.
    3. Identify the threshold that gives the highest Dice.
    4. Plot Dice vs. threshold curve for visualization.

    Args:
        model: Trained U-Net model.
        val_loader: DataLoader containing validation MRI slices.

    Returns:
        best_threshold (float): threshold achieving the highest Dice score.
        thresholds (np.ndarray): list of scanned thresholds.
        dice_scores (np.ndarray): Dice score corresponding to each threshold.
    """

    model.eval()
    thresholds = np.linspace(0.05, 0.95, 20)
    dice_scores = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for th in tqdm(thresholds, desc="Scanning thresholds"):
            dices = []

            for x, y, _, _ in val_loader:
                x = x.to(device)
                y = y.to(device)

                # Forward pass
                logits = model(x)
                probs = torch.sigmoid(logits)
                pred_bin = (probs > th).float()

                # Binarize predictions at this threshold
                pred = (probs > th).float()
                dice = dice_score_binary(pred_bin, y)
                dices.append(dice.item())

            dice_scores.append(np.mean(dices))

    dice_scores = np.array(dice_scores)

    # Find best threshold
    best_idx = np.argmax(dice_scores)
    best_th = float(thresholds[best_idx])

    # ---- Plot Dice vs threshold ----
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, dice_scores, marker="o")
    plt.title("U-Net Validation Dice vs. Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Dice Score")
    plt.grid(True)
    plt.show()

    print(f"Best threshold: {best_th:.3f} | Dice: {dice_scores[best_idx]:.4f}")

    return best_th, thresholds, dice_scores


def undo_resize_and_pad(mask_512: torch.Tensor, orig_h: int, orig_w: int,
                        target_size: int = 512) -> np.ndarray:
    """
    Invert resize_and_pad for a predicted mask.

    mask_512: tensor (H', W') or (1, H', W') with H'=W'=target_size
    returns: numpy (orig_h, orig_w)
    """
    if mask_512.dim() == 2:
        mask = mask_512.unsqueeze(0)  # (1,H,W)
    else:
        mask = mask_512  # assume (1,H,W)

    Ht, Wt = target_size, target_size

    # recompute same scale & padding as in resize_and_pad
    scale = target_size / max(orig_h, orig_w)
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))

    pad_h = Ht - new_h
    pad_w = Wt - new_w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # crop out padding
    mask_cropped = mask[
        :,
        pad_top : Ht - pad_bottom,
        pad_left : Wt - pad_right,
    ]  # (1, new_h, new_w)

    # resize back to original size
    mask_resized = F.interpolate(
        mask_cropped.unsqueeze(0),  # (1,1,new_h,new_w)
        size=(orig_h, orig_w),
        mode="nearest"
    ).squeeze().cpu().numpy()       # (orig_h, orig_w)

    return mask_resized


def predict_and_plot_grid(
    model,
    test_df: pd.DataFrame,
    num_patients: int = 3,
    num_slices: int = 5,
    threshold: float = 0.95,
):
    """
    Plot a grid of predictions:
      rows   = random patients
      cols   = consecutive slices around the middle for each patient

    model    : trained UNet (n_channels=1, n_classes=1)
    test_df  : DataFrame with columns ['patient_id', 'slice_index', 'img', 'orig_h', 'orig_w']
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # make sure df index corresponds to dataset index
    test_df = test_df.reset_index(drop=True)
    dataset = MRITestDataset(test_df, target_size=512)

    unique_pids = test_df["patient_id"].unique()
    num_patients = min(num_patients, len(unique_pids))
    chosen_pids = np.random.choice(unique_pids, size=num_patients, replace=False)

    # prepare figure
    fig, axes = plt.subplots(num_patients, num_slices, figsize=(num_slices * 4, num_patients * 4))
    if num_patients == 1:
        axes = np.expand_dims(axes, axis=0)  # ensure 2D indexing

    window_half = num_slices // 2

    with torch.no_grad():
        for row_idx, pid in enumerate(chosen_pids):
            # all slices for this patient, sorted by slice_index
            grp = test_df[test_df["patient_id"] == pid].sort_values("slice_index")
            grp_indices = grp.index.to_numpy()
            n_slices = len(grp_indices)

            if n_slices == 0:
                continue

            mid_pos = n_slices // 2  # index within this patient's stack

            # choose a window of `num_slices` around the middle
            start = max(0, mid_pos - window_half)
            end = start + num_slices
            if end > n_slices:
                end = n_slices
                start = max(0, end - num_slices)

            # these are global indices into test_df / dataset
            slice_indices_global = grp_indices[start:end]

            # if there are still fewer than num_slices (very small volume),
            # pad by reusing last index
            if len(slice_indices_global) < num_slices:
                last_idx = slice_indices_global[-1]
                slice_indices_global = np.concatenate(
                    [slice_indices_global,
                     np.full(num_slices - len(slice_indices_global), last_idx, dtype=int)]
                )

            for col_idx, ds_idx in enumerate(slice_indices_global):
                ax = axes[row_idx, col_idx]

                x, orig_slice, pid_ds, sl_idx, orig_h, orig_w = dataset[ds_idx]

                # x: (1,H,W) -> (1,1,512,512)
                x_batch = x.unsqueeze(0).to(device)
                logits = model(x_batch)
                probs = torch.sigmoid(logits)
                mask_512 = (probs > threshold).float()[0, 0]  # (512,512)

                # back to original size
                mask_orig = undo_resize_and_pad(
                    mask_512,
                    int(orig_h),
                    int(orig_w),
                )

                img_orig = orig_slice  # numpy (H,W)

                ax.imshow(img_orig, cmap="gray")
                ax.imshow(mask_orig, alpha=0.4, cmap="Reds")
                ax.set_title(f"PID {int(pid_ds)}, z={int(sl_idx)}", fontsize=10)
                ax.axis("off")

    plt.tight_layout()
    plt.show()
