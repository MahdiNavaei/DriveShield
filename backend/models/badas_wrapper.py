import torch
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import sys
import pandas as pd

# EN: Adjust path to import from the local loader
# FA: تنظیم مسیر برای ایمپورت از لودر محلی
_MODEL_DIR = Path(__file__).parent.resolve()
sys.path.append(str(_MODEL_DIR.parent.parent))

from backend.models.badas_loader import load_badas_model

# EN: Configure basic logging
# FA: پیکربندی لاگینگ پایه
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BadasOpenService:
    """
    EN: High-level service for running BADAS-Open collision prediction locally.
    FA: سرویس سطح‌بالا برای اجرای محلی مدل BADAS-Open جهت پیش‌بینی تصادف.
    """
    def __init__(self, device: Optional[str] = None):
        """
        EN: Initializes the service by loading the BADAS-Open model using the modified local loader.
        FA: سرویس را با بارگذاری مدل BADAS-Open با استفاده از لودر محلی اصلاح‌شده مقداردهی اولیه می‌کند.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = load_badas_model(device=device)

    def predict_video(self, video_path: str) -> Dict[str, Any]:
        """
        EN: Run BADAS-Open on a dashcam video and return a structured prediction.
        FA: اجرای BADAS-Open روی یک ویدیوی داش‌کم و برگرداندن خروجی ساخت‌یافته.
        """
        logging.info(f"EN: Running inference on video: {video_path} / FA: در حال اجرای اینفرانس روی ویدیو: {video_path}")
        
        per_frame_probabilities = self.model.predict(video_path)
        
        probabilities_list = per_frame_probabilities.tolist()
        # EN: Replace NaN with 0.0 in probabilities list for JSON compliance
        # FA: جایگزینی مقادیر NaN با 0.0 در لیست احتمالات برای سازگاری با JSON
        probabilities_list = [0.0 if pd.isna(p) else p for p in probabilities_list]
        
        num_frames = len(probabilities_list)
        
        # EN: Filter out NaN values from probabilities before processing
        # FA: فیلتر کردن مقادیر NaN از پروبابلتی‌ها قبل از پردازش
        valid_probs = [p for p in probabilities_list if not pd.isna(p)]
        
        threshold = 0.8
        
        high_risk_indices = [
            i for i, prob in enumerate(probabilities_list) if prob is not None and not pd.isna(prob) and prob >= threshold
        ]
        
        # EN: Calculate the maximum probability from valid probabilities
        # FA: محاسبه بیشترین پروبابلتی از پروبابلتی‌های معتبر
        max_probability = max(valid_probs) if valid_probs else 0.0

        result = {
            "video_path": video_path,
            "num_frames": num_frames,
            "probabilities": probabilities_list,
            "max_probability": max_probability,
            "high_risk_indices": high_risk_indices,
            "threshold": threshold,
        }
        
        logging.info("EN: Inference completed. / FA: اینفرانس کامل شد.")
        return result
