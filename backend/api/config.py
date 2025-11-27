from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class ApiSettings(BaseSettings):
    """
    EN: Configuration options for the DriveShield API service.
    FA: تنظیمات پیکربندی سرویس API پروژه DriveShield.
    """
    api_title: str = "DriveShield Collision Prediction API"
    api_version: str = "0.1.0"
    badas_device: Optional[str] = None  # EN: "cuda", "cpu", or None for auto / FA: "cuda"، "cpu"، یا None برای حالت خودکار
    temp_dir: str = "tmp_videos"

    class Config:
        # EN: This allows loading settings from a .env file, for example.
        # FA: این امکان بارگذاری تنظیمات از یک فایل .env را فراهم می‌کند.
        env_file = ".env"
        env_file_encoding = 'utf-8'

@lru_cache()
def get_settings() -> ApiSettings:
    """
    EN: Return a singleton instance of API settings.
        Using lru_cache ensures the settings are loaded only once.
    FA: یک نمونه singleton از تنظیمات API را برمی‌گرداند.
        استفاده از lru_cache تضمین می‌کند که تنظیمات فقط یک بار بارگذاری شوند.
    """
    return ApiSettings()
