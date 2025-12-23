import torch
import numpy as np
from scipy.ndimage import label
from pathlib import Path
import pandas as pd
import nibabel as nib
import re
from tqdm.auto import tqdm

from src.preprocessing import resize_and_pad
from src.eval import undo_resize_and_pad
from src.utils import check_axial_orientation


def predict_breast_mask_3d(
    model,
    scan_data: np.ndarray,   # (H, W, Z)
    device: torch.device,
    threshold: float,
    target_size: int = 512,
) -> np.ndarray:
    model.eval()
    model.to(device)

    H, W, Z = scan_data.shape
    mask_3d = np.zeros((H, W, Z), dtype=bool)

    with torch.no_grad():
        for z in range(Z):
            sl = scan_data[:, :, z].astype(np.float32)

            # same clipping as in MRIDataset
            p1 = np.percentile(sl, 1)
            p99 = np.percentile(sl, 99)
            sl = np.clip(sl, p1, p99)

            # (H,W) -> (1,H,W) for resize_and_pad
            x = torch.from_numpy(sl).float().unsqueeze(0)  # (1,H,W)

            # resize + pad (expects 3D)
            x = resize_and_pad(x, target_size=target_size, mode="bilinear")  # (1,512,512)

            # normalize like in train MRIDataset
            mean = x.mean()
            std = x.std()
            if std < 1e-6:
                std = 1.0
            x = (x - mean) / std

            # add batch dimension for model
            x = x.unsqueeze(0).to(device)  # (1,1,512,512)

            logits = model(x)              # (1,1,512,512)
            probs = torch.sigmoid(logits)
            mask_512 = (probs > threshold).float()[0, 0]  # (512,512)

            # back to original slice size
            mask_sl = undo_resize_and_pad(mask_512, H, W) > 0.5
            mask_3d[:, :, z] = mask_sl

    return mask_3d


def detect_tumor_candidate(
    scan_data: np.ndarray,          # (H,W,Z), original intensities
    breast_mask: np.ndarray,        # (H,W,Z), bool
    breast_intensity_percentile: float = 98.0,
) -> tuple[np.ndarray, dict]:
    """
    Within the breast mask, find high-intensity voxels and keep the largest
    3D connected component as 'tumor candidate'.

    Returns:
        tumor_mask: 3D bool array (H,W,Z)
        features: dict with simple quantitative features
    """
    # restrict to breast
    breast_pixels = scan_data[breast_mask]
    if breast_pixels.size == 0:
        return np.zeros_like(breast_mask, dtype=bool), {
            "volume_voxels": 0,
            "mean_intensity": np.nan,
            "max_intensity": np.nan,
        }

    thr = np.percentile(breast_pixels, breast_intensity_percentile)

    high_int = (scan_data >= thr) & breast_mask

    labeled, num = label(high_int)
    if num == 0:
        return np.zeros_like(breast_mask, dtype=bool), {
            "volume_voxels": 0,
            "mean_intensity": np.nan,
            "max_intensity": np.nan,
        }

    # choose largest connected component
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # background label 0
    tumor_label = sizes.argmax()
    tumor_mask = (labeled == tumor_label)

    # simple features
    volume_voxels = int(tumor_mask.sum())
    mean_intensity = float(scan_data[tumor_mask].mean())
    max_intensity = float(scan_data[tumor_mask].max())

    features = {
        "volume_voxels": volume_voxels,
        "mean_intensity": mean_intensity,
        "max_intensity": max_intensity,
    }
    return tumor_mask, features


def run_pipeline_on_test(
    model,
    test_scans_folder: Path,
    device: torch.device,
    threshold: float,
) -> pd.DataFrame:
    """
    Breast segmentation -> tumor detection -> feature extraction
    for all test scans. Returns a DataFrame with one row per patient.
    """
    test_scans_folder = Path(test_scans_folder)
    scan_files = sorted(
        list(test_scans_folder.glob("*.nii")) +
        list(test_scans_folder.glob("*.nii.gz"))
    )
    rows = []

    for scan_path in tqdm(scan_files):
        m = re.search(r"(\d+)", scan_path.name)
        if m is None:
            raise ValueError(f"Could not find patient ID in {scan_path.name}")
        pid = int(m.group(1))

        img = nib.load(str(scan_path))
        scan_data = img.get_fdata()
        scan_data = check_axial_orientation(scan_data)  # same as training
        H, W, Z = scan_data.shape

        # breast mask
        breast_mask = predict_breast_mask_3d(
            model,
            scan_data,
            device=device,
            threshold=threshold,
            target_size=512,
        )

        # clean-up: keep largest CC in breast mask
        labeled_b, num_b = label(breast_mask)
        if num_b > 0:
            sizes_b = np.bincount(labeled_b.ravel())
            sizes_b[0] = 0
            main_label = sizes_b.argmax()
            breast_mask = (labeled_b == main_label)

        # tumor candidate
        tumor_mask, feats = detect_tumor_candidate(scan_data, breast_mask)

        feats["patient_id"] = pid
        rows.append(feats)

    feat_df = pd.DataFrame(rows).set_index("patient_id").sort_index()
    return feat_df