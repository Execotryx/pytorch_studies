import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import torch

from unet_vgg19 import UNetVGG19

try:
    import albumentations as A
except Exception as e:  # pragma: no cover
    raise ImportError(
        "This inference script requires albumentations. Install it with:\n"
        "  pip install albumentations\n"
        f"Original import error:\n{e}"
    )

# Optional Markdown-to-console renderer (best UX). Fallback is plain Markdown text.
_RICH_AVAILABLE = False
try:
    from rich.console import Console
    from rich.markdown import Markdown

    _RICH_AVAILABLE = True
    _RICH_CONSOLE = Console()
except Exception:
    _RICH_AVAILABLE = False
    _RICH_CONSOLE = None


def _to_path(p: str | Path) -> Path:
    return Path(os.path.expandvars(str(p))).expanduser()


def resolve_path(p: str | Path, base_dir: Path) -> Path:
    pth = _to_path(p)
    if not pth.is_absolute():
        pth = base_dir / pth
    return pth.resolve()


def make_inference_transform(image_size: int, mean: List[float], std: List[float]) -> A.Compose:
    mean_t: Tuple[float, float, float] = (float(mean[0]), float(mean[1]), float(mean[2]))
    std_t: Tuple[float, float, float] = (float(std[0]), float(std[1]), float(std[2]))

    transforms: list[Any] = [
        A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=(0, 0, 0),
            fill_mask=0,
            position="center",
        ),
        A.Normalize(mean=mean_t, std=std_t, max_pixel_value=255.0),
    ]
    return A.Compose(transforms, is_check_shapes=False)


def _read_image_rgb(image_path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to load image from {image_path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _read_mask_gray(mask_path: Path) -> np.ndarray:
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise ValueError(f"Failed to load mask from {mask_path}")
    if m.ndim == 3:
        m = m[..., 0]
    return m


def _binarize_mask(mask_gray: np.ndarray, mode: str) -> np.ndarray:
    """
    Returns uint8 mask in {0,1}.
    mode:
      - "binary_threshold": (mask > 127)
      - "oxford_trimaps": Oxford-IIIT Pet mapping: foreground=(mask!=2) for trimap values {1,2,3}
    """
    if mode == "oxford_trimaps":
        # foreground = {1,3}, background = 2
        return (mask_gray != 2).astype(np.uint8)
    # default
    return (mask_gray > 127).astype(np.uint8)


def preprocess_image(
    image_path: Path,
    transform: A.Compose,
    mask_path: Optional[Path] = None,
    mask_mode: str = "binary_threshold",
) -> Tuple[torch.Tensor, np.ndarray, Optional[torch.Tensor]]:
    """
    Returns:
      image_t: (1,3,H,W) float tensor
      original_rgb: original RGB image (H,W,3) uint8
      mask_t: (1,1,H,W) float tensor in {0,1} if mask_path provided
    """
    original = _read_image_rgb(image_path)

    if mask_path is not None:
        mask_gray = _read_mask_gray(mask_path)
        mask_bin = _binarize_mask(mask_gray, mode=mask_mode)
        out = transform(image=original, mask=mask_bin)
        image_np = out["image"]
        mask_np = out["mask"].astype(np.uint8)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_np)).float().unsqueeze(0).unsqueeze(0)
    else:
        out = transform(image=original)
        image_np = out["image"]
        mask_t = None

    image_t = torch.from_numpy(np.ascontiguousarray(image_np)).permute(2, 0, 1).contiguous().unsqueeze(0)
    return image_t, original, mask_t


def preprocess_batch(
    image_paths: List[Path],
    transform: A.Compose,
    mask_dir: Optional[Path],
    mask_mode: str,
) -> Tuple[torch.Tensor, List[np.ndarray], List[Path], Optional[torch.Tensor]]:
    batch_tensors: List[torch.Tensor] = []
    originals: List[np.ndarray] = []
    kept_paths: List[Path] = []
    mask_tensors: List[torch.Tensor] = []

    have_masks = mask_dir is not None

    for p in image_paths:
        mp: Optional[Path] = None
        if have_masks:
            # match by stem; accept common mask extensions
            candidates = [
                mask_dir / f"{p.stem}.png",
                mask_dir / f"{p.stem}.jpg",
                mask_dir / f"{p.stem}.jpeg",
                mask_dir / f"{p.stem}.bmp",
                mask_dir / f"{p.stem}.tif",
                mask_dir / f"{p.stem}.tiff",
            ]
            mp = next((c for c in candidates if c.exists()), None)
            if mp is None:
                # fallback: any extension
                any_cands = list(mask_dir.glob(p.stem + ".*"))
                mp = any_cands[0] if any_cands else None
            if mp is None:
                raise FileNotFoundError(f"Mask not found for image: {p.name} in {mask_dir}")

        t, orig, mt = preprocess_image(p, transform, mask_path=mp, mask_mode=mask_mode)
        batch_tensors.append(t)
        originals.append(orig)
        kept_paths.append(p)
        if have_masks:
            assert mt is not None
            mask_tensors.append(mt)

    batch = torch.cat(batch_tensors, dim=0)

    if have_masks:
        masks = torch.cat(mask_tensors, dim=0)  # (B,1,H,W)
        return batch, originals, kept_paths, masks
    return batch, originals, kept_paths, None


def run_inference(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    channels_last: bool = False,
) -> torch.Tensor:
    model.eval()
    images = images.to(device=device, non_blocking=True)

    if channels_last and device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)

    amp_enabled = bool(use_amp and device.type == "cuda")

    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits = model(images)
    return logits


def postprocess_binary(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    pred = (prob >= threshold).to(torch.uint8) * 255
    return pred.squeeze(1).cpu()


def save_mask_png(mask_hw: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), mask_hw)


# -----------------------------
# Validation metrics + Markdown table
# -----------------------------
def _dice_iou_acc_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    eps: float = 1e-6,
) -> Tuple[float, float, float]:
    """
    logits: (B,1,H,W)
    targets: (B,1,H,W) float in {0,1}
    """
    probs = torch.sigmoid(logits.float())
    preds = (probs >= threshold).float()

    # compute in float32 for stability
    preds_f = preds.float()
    targ_f = targets.float()

    inter = (preds_f * targ_f).sum(dim=(2, 3), dtype=torch.float32)
    sum_p = preds_f.sum(dim=(2, 3), dtype=torch.float32)
    sum_t = targ_f.sum(dim=(2, 3), dtype=torch.float32)

    dice = (2.0 * inter + eps) / (sum_p + sum_t + eps)
    union = sum_p + sum_t - inter
    iou = (inter + eps) / (union + eps)

    acc = (preds_f == targ_f).float().mean(dim=(1, 2, 3))

    dice_m = float(torch.nan_to_num(dice, nan=0.0, posinf=0.0, neginf=0.0).mean().item())
    iou_m = float(torch.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0).mean().item())
    acc_m = float(torch.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0).mean().item())
    return dice_m, iou_m, acc_m


def _make_markdown_table(rows: List[Dict[str, Any]], columns: List[Tuple[str, str]]) -> str:
    headers = [h for _, h in columns]
    sep = ["---"] * len(columns)

    def fmt_val(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines: List[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for r in rows:
        line = "| " + " | ".join(fmt_val(r.get(k, "")) for k, _ in columns) + " |"
        lines.append(line)
    return "\n".join(lines)


def _print_markdown(md: str) -> None:
    if _RICH_AVAILABLE and _RICH_CONSOLE is not None:
        _RICH_CONSOLE.print(Markdown(md))
    else:
        print(md)


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint")
    ap.add_argument("--input", type=str, required=True, help="Input image path or directory")
    ap.add_argument("--output", type=str, required=True, help="Output directory for predicted masks")
    ap.add_argument("--image-size", type=int, default=256)

    # ✅ Configurable threshold
    ap.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for binary mask")

    # ✅ Optional validation (if masks provided)
    ap.add_argument(
        "--masks",
        type=str,
        default="",
        help="Optional masks directory (filenames matched by stem). If set, prints Dice/IoU/Acc table.",
    )
    ap.add_argument(
        "--mask-mode",
        type=str,
        default="binary_threshold",
        choices=["binary_threshold", "oxford_trimaps"],
        help="How to binarize masks for validation: thresholded grayscale or Oxford trimaps mapping.",
    )
    ap.add_argument(
        "--table-window",
        type=int,
        default=25,
        help="How many last per-image rows to show in the Markdown table (0 = show all).",
    )

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pretrained-encoder", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--amp-dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--channels-last", action="store_true")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    ckpt_path = resolve_path(args.ckpt, base_dir)
    input_path = resolve_path(args.input, base_dir)
    out_dir = resolve_path(args.output, base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = make_inference_transform(args.image_size, mean, std)

    model = UNetVGG19(
        num_classes=1,
        pretrained=bool(args.pretrained_encoder),
        bilinear=True,
        use_checkpoint=False,
        norm="group",
        gn_groups=16,
        bridge_kernel_size=1,
    ).to(device)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    if input_path.is_dir():
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        image_paths = sorted([p for p in input_path.iterdir() if p.suffix.lower() in exts])
    else:
        image_paths = [input_path]

    if not image_paths:
        raise RuntimeError(f"No images found at: {input_path}")

    mask_dir: Optional[Path] = None
    if str(args.masks).strip():
        mask_dir = resolve_path(args.masks, base_dir)
        if not mask_dir.exists() or not mask_dir.is_dir():
            raise FileNotFoundError(f"--masks must be an existing directory. Got: {mask_dir}")

    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    bs = max(1, int(args.batch_size))
    threshold = float(args.threshold)

    # Validation history (per-image rows)
    val_rows: List[Dict[str, Any]] = []
    running_sum = {"dice": 0.0, "iou": 0.0, "acc": 0.0}
    n_batches = 0

    for i in range(0, len(image_paths), bs):
        batch_paths = image_paths[i : i + bs]

        batch, _originals, kept, batch_masks = preprocess_batch(
            batch_paths, transform, mask_dir=mask_dir, mask_mode=str(args.mask_mode)
        )

        logits = run_inference(
            model=model,
            images=batch,
            device=device,
            use_amp=bool(args.amp),
            amp_dtype=amp_dtype,
            channels_last=bool(args.channels_last),
        )

        # Save predictions
        masks_png = postprocess_binary(logits, threshold=threshold).numpy()
        for p, m in zip(kept, masks_png):
            save_path = out_dir / f"{p.stem}_mask.png"
            save_mask_png(m, save_path)

        # Optional validation
        if batch_masks is not None:
            batch_masks = batch_masks.to(device=device, non_blocking=True)
            d, j, a = _dice_iou_acc_from_logits(logits, batch_masks, threshold=threshold)
            running_sum["dice"] += d
            running_sum["iou"] += j
            running_sum["acc"] += a
            n_batches += 1

            # per-image (approx): compute per-image too for more informative table
            probs = torch.sigmoid(logits.float())
            preds = (probs >= threshold).float()
            targ = batch_masks.float()
            eps = 1e-6

            inter = (preds * targ).sum(dim=(2, 3), dtype=torch.float32)
            sum_p = preds.sum(dim=(2, 3), dtype=torch.float32)
            sum_t = targ.sum(dim=(2, 3), dtype=torch.float32)
            dice = (2.0 * inter + eps) / (sum_p + sum_t + eps)
            union = sum_p + sum_t - inter
            iou = (inter + eps) / (union + eps)
            acc = (preds == targ).float().mean(dim=(1, 2, 3))

            dice = torch.nan_to_num(dice, nan=0.0, posinf=0.0, neginf=0.0).cpu().tolist()
            iou = torch.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0).cpu().tolist()
            acc = torch.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0).cpu().tolist()

            for pth, di, io, ac in zip(kept, dice, iou, acc):
                val_rows.append(
                    {
                        "image": pth.name,
                        "dice": float(di),
                        "iou": float(io),
                        "acc": float(ac),
                    }
                )

        # free cache between batches if desired
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Saved predictions to: {out_dir}")

    # Print validation table if masks were provided
    if mask_dir is not None:
        mean_dice = running_sum["dice"] / max(1, n_batches)
        mean_iou = running_sum["iou"] / max(1, n_batches)
        mean_acc = running_sum["acc"] / max(1, n_batches)

        # show last N rows (or all)
        window = int(args.table_window)
        rows_view = val_rows if window == 0 else val_rows[-window:]

        # append summary row
        rows_with_summary = list(rows_view)
        rows_with_summary.append(
            {
                "image": "**MEAN**",
                "dice": float(mean_dice),
                "iou": float(mean_iou),
                "acc": float(mean_acc),
            }
        )

        cols = [
            ("image", "Image"),
            ("dice", "Dice"),
            ("iou", "IoU"),
            ("acc", "Acc"),
        ]
        md = _make_markdown_table(rows_with_summary, cols)

        print()
        print(f"Validation (threshold={threshold:.3f}, mask_mode={args.mask_mode})")
        _print_markdown(md)


if __name__ == "__main__":
    main()
