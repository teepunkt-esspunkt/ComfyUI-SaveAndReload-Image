# ComfyUI/custom_nodes/ComfyUI-SaveAndReloadImage/__init__.py
import os
import numpy as np
import torch
from PIL import Image

def to_pil(img):
    """
    Accepts torch.Tensor shaped:
      - (B,H,W,3)  BHWC float 0..1 (Comfy IMAGE standard)
      - (H,W,3)    HWC float 0..1
      - (B,C,H,W)  BCHW float 0..1
      - (C,H,W)    CHW float 0..1
    Returns a PIL RGB image of the FIRST item.
    """
    t = img
    if isinstance(t, list):  # some nodes output lists
        t = t[0]

    if isinstance(t, torch.Tensor):
        # pick first in batch if present
        if t.dim() == 4:
            # BHWC or BCHW?
            if t.shape[-1] in (3, 4):  # BHWC
                t = t[0]  # (H,W,C)
            else:  # BCHW
                t = t[0].permute(1, 2, 0)  # (C,H,W) -> (H,W,C)
        elif t.dim() == 3:
            # CHW -> HWC
            if t.shape[0] in (1, 3, 4):
                t = t.permute(1, 2, 0)
        else:
            raise ValueError(f"Unsupported tensor shape: {t.shape}")

        arr = (t.clamp(0, 1) * 255.0).byte().cpu().numpy()
    else:
        raise ValueError("IMAGE must be a torch.Tensor")

    # If grayscale, expand to RGB
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]  # drop alpha for diffusion ref

    return Image.fromarray(arr, mode="RGB")

def from_pil(pil):
    """
    Return Comfy IMAGE as BHWC float 0..1 with batch=1.
    """
    pil = pil.convert("RGB")
    arr = np.array(pil)  # H,W,3
    t = torch.from_numpy(arr).float() / 255.0  # H,W,3
    t = t.unsqueeze(0)  # 1,H,W,3  (BHWC)
    return t

class SaveAndReloadImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "temp_folder": ("STRING", {"default": "output/temp"}),
                "filename": ("STRING", {"default": "bgstrip.png"}),
                "also_save_perm": ("BOOLEAN", {"default": False}),
                "perm_folder": ("STRING", {"default": "output/saved"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("reloaded_image",)
    FUNCTION = "run"
    CATEGORY = "IO"

    def run(self, image, temp_folder, filename, also_save_perm, perm_folder):
        # resolve paths relative to ComfyUI root
        comfy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        temp_folder = os.path.abspath(os.path.join(comfy_root, temp_folder))
        perm_folder = os.path.abspath(os.path.join(comfy_root, perm_folder))
        os.makedirs(temp_folder, exist_ok=True)
        if also_save_perm:
            os.makedirs(perm_folder, exist_ok=True)

        root, ext = os.path.splitext(filename)
        if not ext:
            ext = ".png"
        filename = root + ext

        # save to temp (always overwrite)
        pil = to_pil(image)
        temp_path = os.path.join(temp_folder, filename)
        pil.save(temp_path)

        # optional permanent copy
        if also_save_perm:
            pil.save(os.path.join(perm_folder, filename))

        # reload -> BHWC float
        pil2 = Image.open(temp_path)
        out_img = from_pil(pil2)
        return (out_img,)

NODE_CLASS_MAPPINGS = {"SaveAndReloadImage": SaveAndReloadImage}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveAndReloadImage": "Save & Reload Image"}
