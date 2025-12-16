import argparse
import os
from pathlib import Path
from typing import Any, List, Tuple

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


def preprocess_image(image_path: Path, transform: A.Compose) -> Tuple[torch.Tensor, np.ndarray]:
    original = _read_image_rgb(image_path)
    out = transform(image=original)
    image_np = out["image"]
    image_t = torch.from_numpy(np.ascontiguousarray(image_np)).permute(2, 0, 1).contiguous()
    image_t = image_t.unsqueeze(0)
    return image_t, original


def preprocess_batch(image_paths: List[Path], transform: A.Compose) -> Tuple[torch.Tensor, List[np.ndarray], List[Path]]:
    batch_tensors: List[torch.Tensor] = []
    originals: List[np.ndarray] = []
    kept_paths: List[Path] = []

    for p in image_paths:
        t, orig = preprocess_image(p, transform)
        batch_tensors.append(t)
        originals.append(orig)
        kept_paths.append(p)

    batch = torch.cat(batch_tensors, dim=0)
    return batch, originals, kept_paths


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


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint")
    ap.add_argument("--input", type=str, required=True, help="Input image path or directory")
    ap.add_argument("--output", type=str, required=True, help="Output directory for predicted masks")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--threshold", type=float, default=0.5)
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

    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    bs = max(1, int(args.batch_size))

    for i in range(0, len(image_paths), bs):
        batch_paths = image_paths[i : i + bs]
        batch, _originals, kept = preprocess_batch(batch_paths, transform)

        logits = run_inference(
            model=model,
            images=batch,
            device=device,
            use_amp=bool(args.amp),
            amp_dtype=amp_dtype,
            channels_last=bool(args.channels_last),
        )
        masks = postprocess_binary(logits, threshold=float(args.threshold)).numpy()

        for p, m in zip(kept, masks):
            save_path = out_dir / f"{p.stem}_mask.png"
            save_mask_png(m, save_path)

    print(f"Saved predictions to: {out_dir}")


if __name__ == "__main__":
    main()
