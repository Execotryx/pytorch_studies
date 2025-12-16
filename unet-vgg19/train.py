import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import Dataset, DataLoader

from unet_vgg19 import UNetVGG19

try:
    import albumentations as A
except Exception as e:
    raise ImportError(
        "This training script requires albumentations. Install it with:\n"
        "  pip install albumentations\n"
        f"Original import error:\n{e}"
    )

try:
    from torchvision.datasets import OxfordIIITPet
except Exception as e:
    raise ImportError(
        "This training script requires torchvision. Install it with:\n"
        "  pip install torchvision\n"
        f"Original import error:\n{e}"
    )


# -----------------------------
# Cross-platform path helpers
# -----------------------------
def _to_path(p: str | Path) -> Path:
    return Path(os.path.expandvars(str(p))).expanduser()


def resolve_path(p: str | Path, base_dir: Path) -> Path:
    pth = _to_path(p)
    if not pth.is_absolute():
        pth = base_dir / pth
    return pth.resolve()


def ensure_dir(path: str | Path) -> Path:
    p = _to_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _json_default(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, torch.device):
        return str(o)
    if isinstance(o, torch.dtype):
        return str(o)
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path_p = _to_path(path)
    ensure_dir(path_p.parent)
    with open(path_p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def configure_logger(save_dir: str | Path) -> logging.Logger:
    save_dir_p = ensure_dir(save_dir)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(str(save_dir_p / "train.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def configure_tf32(enable: bool) -> None:
    if not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.fp32_precision = "tf32" if enable else "ieee"
    conv_backend = getattr(torch.backends.cudnn, "conv", None)
    if conv_backend is not None:
        setattr(conv_backend, "fp32_precision", "tf32" if enable else "ieee")


# -----------------------------
# Dataset (custom folder)
# -----------------------------
class SegDataset(Dataset):
    """
    Custom dataset:
      root/
        train/images/*
        train/masks/*
        val/images/*
        val/masks/*
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        num_classes: int,
        transform: Optional[A.Compose] = None,
    ) -> None:
        super().__init__()
        self.root = _to_path(root)
        self.split = split
        self.num_classes = num_classes
        self.transform = transform

        self.images_dir = self.root / split / "images"
        self.masks_dir = self.root / split / "masks"

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")

        self.image_paths = sorted(
            [
                p
                for p in self.images_dir.iterdir()
                if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
            ]
        )
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in: {self.images_dir}")

        self.has_masks = self.masks_dir.exists()

    def __len__(self) -> int:
        return len(self.image_paths)

    def _read_image_rgb(self, path: Path) -> np.ndarray:
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Failed to read image: {path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _read_mask(self, path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Failed to read mask: {path}")
        if mask.ndim == 3:
            mask = mask[..., 0]
        return mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_paths[idx]
        image = self._read_image_rgb(img_path)

        sample: Dict[str, Any] = {"image": image, "image_path": str(img_path)}

        if self.has_masks:
            mask_path = self.masks_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                candidates = list(self.masks_dir.glob(img_path.stem + ".*"))
                if not candidates:
                    raise FileNotFoundError(f"Mask not found for image: {img_path.name}")
                mask_path = candidates[0]
            mask = self._read_mask(mask_path)

            if self.num_classes == 1:
                mask = (mask > 127).astype(np.uint8)

            sample["mask"] = mask

        if self.transform is not None:
            if self.has_masks:
                aug = self.transform(image=sample["image"], mask=sample["mask"])
                sample["image"] = aug["image"]
                sample["mask"] = aug["mask"]
            else:
                aug = self.transform(image=sample["image"])
                sample["image"] = aug["image"]

        image_t = torch.from_numpy(np.ascontiguousarray(sample["image"])).permute(2, 0, 1).contiguous()
        sample["image_t"] = image_t

        if self.has_masks:
            mask_np = sample["mask"]
            if self.num_classes == 1:
                mask_t = torch.from_numpy(np.ascontiguousarray(mask_np)).float().unsqueeze(0)
            else:
                mask_t = torch.from_numpy(np.ascontiguousarray(mask_np)).long()
            sample["mask_t"] = mask_t

        return sample


# -----------------------------
# Dataset (Torchvision Oxford-IIIT Pet, auto-download)
# -----------------------------
class TorchvisionSegWrapper(Dataset):
    """
    OxfordIIITPet target_types="segmentation":
      trimaps contain values {1,2,3}; background is 2.
      Binary mapping: (mask != 2) -> {0,1}.
    """

    def __init__(
        self,
        base: Dataset,
        indices: List[int],
        num_classes: int,
        transform: Optional[A.Compose],
    ) -> None:
        super().__init__()
        self.base = base
        self.indices = indices
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        idx = self.indices[i]
        img_pil, mask_pil = self.base[idx]

        image = np.array(img_pil.convert("RGB"))
        mask = np.array(mask_pil)
        if mask.ndim == 3:
            mask = mask[..., 0]

        if self.num_classes == 1:
            mask = (mask != 2).astype(np.uint8)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        image_t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).contiguous()
        if self.num_classes == 1:
            mask_t = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        else:
            mask_t = torch.from_numpy(np.ascontiguousarray(mask)).long()

        return {"image_t": image_t, "mask_t": mask_t, "image_path": str(idx)}


# -----------------------------
# Albumentations transforms
# -----------------------------
def make_albu_transforms(image_size: int, mean: List[float], std: List[float], train: bool) -> A.Compose:
    mean_t: Tuple[float, float, float] = (float(mean[0]), float(mean[1]), float(mean[2]))
    std_t: Tuple[float, float, float] = (float(std[0]), float(std[1]), float(std[2]))

    transforms: List[Any] = [
        A.LongestMaxSize(max_size=image_size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=(0, 0, 0),
            fill_mask=0,
            position="center",
        ),
    ]

    if train:
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Affine(
                    scale=(0.9, 1.1),
                    translate_percent=(-0.05, 0.05),
                    rotate=(-12, 12),
                    shear=(0.0, 0.0),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=(0, 0, 0),
                    fill_mask=0,
                    p=0.7,
                ),
                A.OneOf(
                    [
                        A.ElasticTransform(
                            alpha=18,
                            sigma=5,
                            approximate=False,
                            same_dxdy=False,
                            interpolation=cv2.INTER_LINEAR,
                            mask_interpolation=cv2.INTER_NEAREST,
                            border_mode=cv2.BORDER_CONSTANT,
                            fill=(0, 0, 0),
                            fill_mask=0,
                            p=1.0,
                        ),
                        A.GridDistortion(
                            num_steps=5,
                            distort_limit=0.2,
                            interpolation=cv2.INTER_LINEAR,
                            mask_interpolation=cv2.INTER_NEAREST,
                            border_mode=cv2.BORDER_CONSTANT,
                            fill=(0, 0, 0),
                            fill_mask=0,
                            p=1.0,
                        ),
                    ],
                    p=0.25,
                ),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            ]
        )

    transforms.append(A.Normalize(mean=mean_t, std=std_t, max_pixel_value=255.0))
    return A.Compose(transforms, is_check_shapes=False)


# -----------------------------
# Losses & Metrics (FP32-safe under AMP)
# -----------------------------
def dice_coeff_binary_from_probs(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs_f = probs.float()
    target_f = target.float()

    inter = (probs_f * target_f).sum(dim=(2, 3), dtype=torch.float32)
    union = probs_f.sum(dim=(2, 3), dtype=torch.float32) + target_f.sum(dim=(2, 3), dtype=torch.float32)
    dice = (2.0 * inter + eps) / (union + eps)
    dice = torch.nan_to_num(dice, nan=0.0, posinf=0.0, neginf=0.0)
    return dice.mean()


def iou_binary_from_probs(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs_f = probs.float()
    target_f = target.float()
    inter = (probs_f * target_f).sum(dim=(2, 3), dtype=torch.float32)
    union = (
        probs_f.sum(dim=(2, 3), dtype=torch.float32)
        + target_f.sum(dim=(2, 3), dtype=torch.float32)
        - inter
    )
    iou = (inter + eps) / (union + eps)
    iou = torch.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)
    return iou.mean()


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits.float())
    return 1.0 - dice_coeff_binary_from_probs(probs, target)


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, bce_weight: float = 0.5) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits.float(), target.float())
    dl = dice_loss_from_logits(logits, target)
    return bce_weight * bce + (1.0 - bce_weight) * dl


@torch.no_grad()
def compute_metrics_binary(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits.float())
    pred = (probs >= threshold).float()

    dice = float(dice_coeff_binary_from_probs(pred, target).item())
    iou = float(iou_binary_from_probs(pred, target).item())
    acc = float((pred == target.float()).float().mean().item())
    return {"dice": dice, "iou": iou, "acc": acc}


# -----------------------------
# Encoder freeze schedule
# -----------------------------
def apply_freeze_schedule(model: UNetVGG19, stages_to_train: Iterable[str]) -> None:
    stages_to_train = set(stages_to_train)
    for name, param in model.named_parameters():
        if name.startswith("enc"):
            trainable = any(name.startswith(s) for s in stages_to_train)
            param.requires_grad = trainable
        else:
            param.requires_grad = True


# -----------------------------
# Training loop
# -----------------------------
@dataclass
class TrainState:
    epoch: int = 0
    best_metric: float = -1e9


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype,
    grad_accum_steps: int,
    channels_last: bool,
    logger: logging.Logger,
) -> float:
    model.train()
    running = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        images = batch["image_t"].to(device, non_blocking=True)
        masks = batch["mask_t"].to(device, non_blocking=True)

        if channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            logits = model(images)
            loss = bce_dice_loss(logits, masks) / float(grad_accum_steps)

        if not torch.isfinite(loss):
            logger.warning(f"Non-finite loss at step={step}: {loss.item()}. Skipping step.")
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running += float(loss.item()) * float(grad_accum_steps)

    return running / max(1, len(loader))


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype,
    channels_last: bool,
) -> Dict[str, float]:
    model.eval()
    totals = {"dice": 0.0, "iou": 0.0, "acc": 0.0}
    n = 0

    for batch in loader:
        images = batch["image_t"].to(device, non_blocking=True)
        masks = batch["mask_t"].to(device, non_blocking=True)

        if channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            logits = model(images)

        m = compute_metrics_binary(logits, masks)
        for k in totals:
            totals[k] += m[k]
        n += 1

    return {k: (v / n if n else 0.0) for k, v in totals.items()}


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    config: Dict[str, Any] = {
        "seed": 42,
        "image_size": 256,
        "num_classes": 1,
        "dataset": "oxford_pets",  # "custom" or "oxford_pets"
        "custom_root": "data",
        "download": True,
        "val_fraction": 0.1,
        "pretrained_encoder": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "epochs": 25,
        "batch_size": 4,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "amp": True,
        "amp_dtype": "float16",
        "grad_accum_steps": 1,
        "channels_last": True,
        # ✅ constant LR requested
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "save_best_metric": "iou",
        "enable_tf32": True,
        "decoder_upsample": "bilinear",
        "save_dir": "checkpoints",
        "freeze_schedule": {
            "enabled": True,
            "stages": [
                {"epoch": 0, "stages": []},
                {"epoch": 3, "stages": ["enc5"]},
                {"epoch": 6, "stages": ["enc4"]},
                {"epoch": 9, "stages": ["enc3"]},
                {"epoch": 12, "stages": ["enc2"]},
                {"epoch": 15, "stages": ["enc1"]},
            ],
        },
    }

    save_dir = resolve_path(config["save_dir"], base_dir)
    data_root = resolve_path(config["custom_root"], base_dir)
    ensure_dir(save_dir)

    logger = configure_logger(save_dir)
    save_json(config, save_dir / "config.json")

    set_seed(int(config["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Data root: {data_root}")
    logger.info(f"Save dir: {save_dir}")
    logger.info(f"Constant LR: {config['lr']}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        configure_tf32(bool(config.get("enable_tf32", True)))

    amp_dtype = torch.float16 if config["amp_dtype"] == "float16" else torch.bfloat16
    amp_enabled = bool(config["amp"]) and device.type == "cuda"

    train_tf = make_albu_transforms(config["image_size"], config["mean"], config["std"], train=True)
    val_tf = make_albu_transforms(config["image_size"], config["mean"], config["std"], train=False)

    ds_name = str(config.get("dataset", "custom")).lower()

    if ds_name == "custom":
        train_ds = SegDataset(data_root, "train", int(config["num_classes"]), transform=train_tf)
        val_ds = SegDataset(data_root, "val", int(config["num_classes"]), transform=val_tf)
        logger.info(f"Custom dataset: train={len(train_ds)} val={len(val_ds)}")

    elif ds_name == "oxford_pets":
        base = OxfordIIITPet(
            root=str(data_root),
            split="trainval",
            target_types="segmentation",
            download=bool(config.get("download", True)),
        )

        n = len(base)
        val_fraction = float(config.get("val_fraction", 0.1))
        n_val = max(1, int(round(n * val_fraction)))

        g = torch.Generator().manual_seed(int(config["seed"]))
        perm = torch.randperm(n, generator=g).tolist()
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        train_ds = TorchvisionSegWrapper(base, train_idx, int(config["num_classes"]), transform=train_tf)
        val_ds = TorchvisionSegWrapper(base, val_idx, int(config["num_classes"]), transform=val_tf)

        logger.info(
            f"Oxford-IIIT Pet: base(trainval)={n} train={len(train_ds)} val={len(val_ds)} "
            f"(download={bool(config.get('download', True))})"
        )
    else:
        raise ValueError(f"Unknown dataset: {config['dataset']}")

    loader_kwargs: Dict[str, Any] = {
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "pin_memory": bool(config["pin_memory"]) and device.type == "cuda",
        "persistent_workers": bool(config["persistent_workers"]) and int(config["num_workers"]) > 0,
        "drop_last": False,
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    bilinear = (str(config.get("decoder_upsample", "bilinear")).lower() != "transpose")
    model = UNetVGG19(
        num_classes=int(config["num_classes"]),
        pretrained=bool(config["pretrained_encoder"]),
        bilinear=bilinear,
        use_checkpoint=False,
        norm="group",
        gn_groups=16,
        bridge_kernel_size=1,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = GradScaler(enabled=amp_enabled)

    freeze_schedule = config["freeze_schedule"]
    schedule_points = freeze_schedule["stages"] if freeze_schedule.get("enabled", False) else []
    schedule_points = sorted(schedule_points, key=lambda x: x["epoch"])

    state = TrainState(epoch=0, best_metric=-1e9)

    for epoch in range(int(config["epochs"])):
        state.epoch = epoch

        if schedule_points:
            stages: List[str] = []
            for sp in schedule_points:
                if epoch >= int(sp["epoch"]):
                    stages = list(sp["stages"])
            apply_freeze_schedule(model, stages_to_train=stages)
            logger.info(f"[epoch {epoch}] Encoder trainable stages: {stages}")

        loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp=amp_enabled,
            amp_dtype=amp_dtype,
            grad_accum_steps=int(config["grad_accum_steps"]),
            channels_last=bool(config["channels_last"]),
            logger=logger,
        )

        metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
            amp=amp_enabled,
            amp_dtype=amp_dtype,
            channels_last=bool(config["channels_last"]),
        )

        logger.info(
            f"Epoch {epoch:03d} | loss={loss:.4f} | dice={metrics['dice']:.4f} | "
            f"iou={metrics['iou']:.4f} | acc={metrics['acc']:.4f}"
        )

        metric_key = str(config["save_best_metric"])
        metric_val = float(metrics.get(metric_key, 0.0))
        if metric_val > state.best_metric:
            state.best_metric = metric_val
            ckpt_path = save_dir / "best.pt"
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, str(ckpt_path))
            logger.info(f"Saved best checkpoint to: {ckpt_path} (best {metric_key}={state.best_metric:.4f})")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("Done.")


if __name__ == "__main__":
    main()
