from pydantic import BaseModel
from typing import List

class StatusResponse(BaseModel):
    """
    EN: Health-check response for the API.
    FA: پاسخ بررسی سلامت سرویس API.
    """
    status: str
    message_en: str
    message_fa: str
    model_loaded: bool

class BadasPredictionSummary(BaseModel):
    """
    EN: Summary information for BADAS-Open prediction.
    FA: اطلاعات خلاصه برای پیش‌بینی BADAS-Open.
    """
    video_filename: str
    num_frames: int
    max_probability: float
    threshold: float
    num_high_risk_frames: int
    high_risk_indices_sample: List[int]
    message_en: str
    message_fa: str

class BadasFullPrediction(BadasPredictionSummary):
    """
    EN: Extended prediction including full probability timeline.
    FA: نسخه توسعه‌یافته پیش‌بینی شامل کل تایم‌لاین پروبابلتی.
    """
    probabilities: List[float]
