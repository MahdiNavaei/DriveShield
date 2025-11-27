import sys
import os
from pathlib import Path
from huggingface_hub import snapshot_download

def load_badas_model(device="cuda"):
    """
    EN: Loads the BADAS-Open model from local files.
    FA: مدل BADAS-Open را از فایل‌های محلی بارگذاری می‌کند.
    """
    # EN: Use local paths instead of downloading
    # FA: استفاده از مسیرهای محلی به جای دانلود کردن
    model_dir = Path(__file__).parent.resolve()
    src_path = model_dir / "src"
    
    # EN: This check is important. If 'src' doesn't exist, we cannot proceed.
    # FA: این بررسی مهم است. اگر 'src' وجود نداشته باشد، نمی‌توانیم ادامه دهیم.
    if not src_path.is_dir():
        # EN: We will download it once if it's missing. This is a setup step.
        # FA: اگر وجود نداشته باشد، یک بار آن را دانلود می‌کنیم. این یک مرحله راه‌اندازی است.
        print("EN: 'src' directory not found. Downloading required source files once... / FA: پوشه 'src' یافت نشد. در حال دانلود فایل‌های سورس مورد نیاز برای یک بار...")
        snapshot_download(repo_id="nexar-ai/nexight", allow_patterns=["src/*"], local_dir=model_dir)

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from models.vjepa import VJEPAModel
    
    checkpoint_path = model_dir / "badas_open.pth"
    config_path = model_dir / "config.json"

    if not checkpoint_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("EN: Ensure badas_open.pth and config.json are in backend/models/ / FA: اطمینان حاصل کنید که badas_open.pth و config.json در backend/models/ قرار دارند")
    
    import json
    with open(config_path, "r") as f:
        config = json.load(f)

    model = VJEPAModel(
        model_name=config.get("model_name", "facebook/vjepa2-vitl-fpc16-256-ssv2"),
        checkpoint_path=str(checkpoint_path),
        frame_count=config.get("frame_count", 16),
        img_size=config.get("img_size", 224),
        window_stride=config.get("window_stride", 1),
        target_fps=config.get("target_fps", 8.0),
        use_sliding_window=config.get("use_sliding_window", True),
    )
    model.load()
    # EN: VJEPAModel handles device internally, no need to call .to()
    # FA: VJEPAModel مدیریت device را به صورت داخلی انجام می‌دهد، نیازی به فراخوانی .to() نیست
    return model
