from pathlib import Path
from dataclasses import dataclass
import nibabel as nib
from typing import Iterable, Tuple
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import random
import matplotlib.pyplot as plt


@dataclass
class DatasetPaths:
    base: Path
    inference_data: Path
    train_folder: Path
    gt_folder: Path
    scans_folder: Path
    clinical_data: Path


def validate_dataset_structure(
    data_location: str | Path,
    verbose: bool = True
) -> DatasetPaths:
    """
    Validate that the dataset directory structure exists and is non-empty.

    Checks the presence and non-emptiness of:
    - inference_data/
    - train_data/
    - train_data/GT/
    - train_data/scans/
    - Clinical_and_Other_Features.xlsx

    Parameters
    ----------
    data_location : str or Path
        Path to the base dataset folder (e.g. 'drive/MyDrive/assignment').
    verbose : bool, default=True
        If True, prints a summary of validated paths and file counts.

    Returns
    -------
    DatasetPaths
        A dataclass instance containing Path objects for all validated folders.

    Raises
    ------
    AssertionError
        If any required directory or file is missing or empty.
    """

    base = Path(data_location)

    inference_data = base / "inference_data"
    train_folder   = base / "train_data"
    gt_folder      = train_folder / "GT"
    scans_folder   = train_folder / "scans"
    clinical_data  = base / "Clinical_and_Other_Features.xlsx"

    # ---- Existence checks ----
    for p in [base, inference_data, train_folder, gt_folder, scans_folder]:
        assert p.exists(), f"Missing folder: {p}"

    assert clinical_data.exists(), f"Missing file: {clinical_data}"

    # ---- Non-empty checks ----
    assert any(inference_data.iterdir()), f"{inference_data} is empty"
    assert any(train_folder.iterdir()),   f"{train_folder} is empty"
    assert any(gt_folder.iterdir()),      f"{gt_folder} is empty"
    assert any(scans_folder.iterdir()),   f"{scans_folder} is empty"

    if verbose:
        print("\n  Dataset structure validated successfully:")
        print(f"  Base folder:          {base}")
        print(f"  Clinical features:    {clinical_data.exists()}")
        print(f"  Inference images:     {len(list(inference_data.iterdir()))} files")
        print(f"  Train folder:         {len(list(train_folder.iterdir()))} items")
        print(f"  ├─ GT masks:          {len(list(gt_folder.iterdir()))} files")
        print(f"  └─ Scans:             {len(list(scans_folder.iterdir()))} files\n")

    return DatasetPaths(
        base=base,
        inference_data=inference_data,
        train_folder=train_folder,
        gt_folder=gt_folder,
        scans_folder=scans_folder,
        clinical_data=clinical_data
    )


def _probe_scan(path_str: str) -> Tuple[str, Tuple[str, str, str], Tuple[float, float, float]]:
    """
    Load one NIfTI and return (filename, orientation, voxel_size).
    This runs in a separate process.
    """
    p = Path(path_str)
    img = nib.load(path_str)
    orient = nib.aff2axcodes(img.affine)
    zooms = img.header.get_zooms()[:3]
    return p.name, orient, zooms



def check_affine_consistency(
    scans_folder: Path,
    max_workers: int | None = None,
) -> None:
    """
    Check that all scans share the same orientation and voxel spacing,
    using multiprocessing + a progress bar.
    """
    scan_files = sorted(scans_folder.glob("*.nii.gz"))
    assert scan_files, f"No .nii.gz files found in {scans_folder}"
    print("Using workers:", max_workers)
    # Submit jobs
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_probe_scan, str(p)) for p in scan_files]

        results = []
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Checking affines",
        ):
            results.append(fut.result())

    # Use the first scan as reference
    ref_name, ref_orient, ref_zooms = results[0]
    print("Reference scan:", ref_name)
    print("  Orientation:", ref_orient)
    print("  Voxel size :", ref_zooms)

    problems = []

    for name, orient, zooms in results[1:]:
        same_orient = orient == ref_orient
        same_zooms = np.allclose(zooms, ref_zooms)

        if not (same_orient and same_zooms):
            problems.append((name, orient, zooms))

    if not problems:
        print("\n All scans have the same orientation and voxel spacing.")
    else:
        print("\n Found scans with different orientation or spacing:")
        for name, orient, zooms in problems:
            print(f"  {name}: orientation={orient}, voxel size={zooms}")


def check_axial_orientation(
    data: np.ndarray,
    verbose: bool = False
) -> np.ndarray:
    """
    Heuristic axial orientation check.

    Steps:
      1. Take the slice at the 90th percentile of the Z axis.
      2. Compare brightness in upper vs lower half of that slice.
      3. If the upper half is brighter, flip along axis 0 (vertical).
      4. Compare mean intensity at 10th vs 90th Z-percentile slices.
      5. If the Z-axis ordering appears reversed, flip along axis 2.

    Args:
        data: 3D volume (X, Y, Z).
        verbose: If True, print diagnostic messages.

    Returns:
        Oriented volume with consistent axial and Z-axis orientation.
    """

    # 1) pick z indices
    z_idx = int(0.9 * (data.shape[2] - 1))
    z_idx_10 = int(0.1 * (data.shape[2] - 1))

    sl = data[:, :, z_idx]

    # 2) split into upper vs lower half along axis 0 (rows)
    mid = sl.shape[0] // 2
    upper = sl[:mid, :]
    lower = sl[mid:, :]

    # 3) measure brightness
    upper_mean = upper.mean()
    lower_mean = lower.mean()

    # 4) vertical flip check
    if upper_mean > lower_mean:
        if verbose:
            print("[Orientation] upper is brighter → flipping axis 0")
        data = data[::-1, :, :]
    else:
        if verbose:
            print("[Orientation] vertical orientation looks OK")

    if data[:, :, z_idx_10].mean() > data[:, :, z_idx].mean():
        if verbose:
            print("[Orientation] Z-axis appears reversed → flipping axis 2")
        data = data[:, :, ::-1]
    else:
        if verbose:
            print("[Orientation] Z-axis orientation looks OK")

    return data



def load_patient_scan(
    scans_folder: Path,
    patient_id: int | str | None = None,
    check_orientation: bool = True,
    verbose: bool = False
) -> tuple[str, np.ndarray]:
    """
    Load one patient's scan as a NumPy array.

    Parameters
    ----------
    scans_folder : Path
        Folder containing .nii.gz scans.
    patient_id : int | str | None
        If None, choose a random scan.
        Otherwise, pick the first file whose name contains this id.

    Returns
    -------
    scan_name : str
        File name of the selected scan.
    data : np.ndarray
        3D volume (result of img.get_fdata()).
    """
    scan_files = sorted(scans_folder.glob("*.nii.gz"))
    assert scan_files, f"No .nii.gz files found in {scans_folder}"

    if patient_id is None:
        scan_path = random.choice(scan_files)
    else:
        patient_id = str(patient_id)
        matches = [f for f in scan_files if patient_id in f.name]
        assert matches, f"No scan file found containing '{patient_id}'"
        scan_path = matches[0]

    print("Showing:", scan_path.name)
    print(scan_path)
    img = nib.load(str(scan_path))
    data = img.get_fdata()

    if check_orientation:
        data = check_axial_orientation(data,verbose=verbose)

    return scan_path.name, data


def plot_percentile_slices(
    data: np.ndarray,
    scan_name: str,
    percents: list[int] | tuple[int, ...] = (10, 25, 50, 75, 90),
) -> None:
    """
    Plot 3×5 slices at given percentiles for a single 3D volume.

    Rows: Sagittal (x), Coronal (y), Axial (z).
    Columns: percentiles along each axis (e.g. 10%, 25%, 50%, 75%, 90%).
    """
    percents = list(percents)

    def percentile_indices(n: int):
        return [int(round(p / 100 * (n - 1))) for p in percents]

    idx_x = percentile_indices(data.shape[0])
    idx_y = percentile_indices(data.shape[1])
    idx_z = percentile_indices(data.shape[2])

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle(scan_name, fontsize=18)

    row_names = ["Sagittal", "Coronal", "Axial"]

    # --- row 0: sagittal (x slices) ---
    for col, x in enumerate(idx_x):
        sl = data[x, :, :]          # (Y, Z)
        ax = axes[0, col]
        ax.imshow(sl.T, cmap="gray")
        ax.set_title(f"x={x} ({percents[col]}%)", fontsize=9)
        ax.axis("off")

    # --- row 1: coronal (y slices) ---
    for col, y in enumerate(idx_y):
        sl = data[:, y, :]          # (X, Z)
        ax = axes[1, col]
        ax.imshow(sl.T, cmap="gray")
        ax.set_title(f"y={y} ({percents[col]}%)", fontsize=9)
        ax.axis("off")

    # --- row 2: axial (z slices) ---
    for col, z in enumerate(idx_z):
        sl = data[:, :, z]          # (X, Y)
        ax = axes[2, col]
        ax.imshow(sl, cmap="gray")
        ax.set_title(f"z={z} ({percents[col]}%)", fontsize=9)
        ax.axis("off")

    # row labels on the left (Sagittal / Coronal / Axial)
    for i, label in enumerate(row_names):
        fig.text(0.02, 0.90 - i * 0.32, label, fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0.04, 0, 1, 0.95])
    plt.show()


def show_overlay(scan, mask, z=None):
    # choose a slice that actually contains mask
    if z is None:
        # pick slice with max mask area
        z = int(np.argmax(mask.sum(axis=(0, 1))))
        print("Using slice", z)

    assert scan.shape == mask.shape

    slice_img = scan[:, :, z]
    slice_mask = mask[:, :, z]

    plt.figure(figsize=(6, 6))
    # base image
    plt.imshow(slice_img, cmap="gray")

    # full mask overlay
    # assume mask is 0/1 or 0/255; turn it into bool for clarity
    mask_bool = slice_mask > 0
    plt.imshow(mask_bool, cmap="Reds", alpha=0.3)

    plt.title(f"Overlay – slice {z}")
    plt.axis("off")
    plt.show()


def orient_mask_up(
    mask: np.ndarray,
    verbose: bool = False
):
    # if mask is not strictly 0/1, treat >0 as foreground
    mask_bin = mask > 0

    # pick axial slice with maximum mask area
    z_max = int(np.argmax(mask_bin.sum(axis=(0, 1))))
    sl = mask_bin[:, :, z_max]

    H, W = sl.shape

    # row sums: how many masked pixels in each horizontal row
    row_counts = sl.sum(axis=1)  # shape (H,)
    dominant_row = int(np.argmax(row_counts))

    # count above/below dominant row (exclude the row itself)
    above_count = int(sl[:dominant_row, :].sum())
    below_count = int(sl[dominant_row + 1 :, :].sum())

    # infer current direction
    # If most of the mask mass is below the dominant row, call it "down"
    if above_count < below_count:
        if verbose:
            print("[Mask orient] mask looks 'down' → flipping along axis 0")
        mask = mask[::-1, :, :]
    else:
        if verbose:
            print("[Mask orient] mask already 'looks up' → no flip")
    return mask


def align_mask_left_right(
    scan: np.ndarray,
    mask: np.ndarray,
    dead_percentile: float = 40.0,
    verbose: bool = False,
) -> np.ndarray:
    """
    Decide whether to flip the mask along axis 1 (left–right) to better match the scan.

    Assumes:
      * scan is already correctly oriented (breasts down).
      * mask is already oriented 'up' using orient_mask_up().

    Args:
        scan: MRI scan volume.
        mask: Binary or soft segmentation mask.
        dead_percentile: Percentile used to define low-intensity background.
        verbose: If True, print diagnostic messages.

    Returns:
        Left–right aligned mask.
    """

    mask_bin = mask > 0

    # 1) slice with maximum mask area
    z_max = int(np.argmax(mask_bin.sum(axis=(0, 1))))

    if verbose:
        print(f"[Mask Y-orient] using slice z={z_max} for scoring")

    scan_slice = scan[:, :, z_max]
    mask_slice_orig = mask_bin[:, :, z_max]
    mask_slice_flip = mask_bin[:, ::-1, z_max]

    # 2) define dead zone in scan (dark background)
    thr = np.percentile(scan_slice, dead_percentile)
    dead_zone = scan_slice < thr

    # 3) compute overlap for original vs flipped-y
    overlap_orig = np.logical_and(mask_slice_orig, dead_zone).sum()
    overlap_flip = np.logical_and(mask_slice_flip, dead_zone).sum()

    if verbose:
        print(f"[Mask Y-orient] original dead-overlap = {overlap_orig}")
        print(f"[Mask Y-orient] flipped-y dead-overlap = {overlap_flip}")

    # 4) choose the version with less overlap
    if overlap_flip < overlap_orig:
        if verbose:
            print("[Mask Y-orient] flipping along axis 1")
        return mask[:, ::-1, :]
    else:
        if verbose:
            print("[Mask Y-orient] keeping original orientation")
        return mask