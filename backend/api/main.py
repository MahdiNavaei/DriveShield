import logging
from pathlib import Path
from typing import Annotated
import os
import sys

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# EN: Adjust sys.path to allow importing from the 'backend' package
# FA: تنظیم sys.path برای امکان وارد کردن ماژول‌ها از پکیج 'backend'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.api.config import get_settings, ApiSettings
from backend.api.schemas import StatusResponse, BadasFullPrediction, BadasPredictionSummary
from backend.api.utils import save_upload_to_temp, safe_remove_file
from backend.models.badas_wrapper import BadasOpenService

# EN: Configure logging
# FA: پیکربندی لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# EN: Initialize settings
# FA: مقداردهی اولیه تنظیمات
settings = get_settings()

# EN: Create the FastAPI app instance
# FA: ساخت نمونه اپلیکیشن FastAPI
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
)

# EN: Add CORS middleware to allow all origins
# FA: اضافه کردن میان‌افزار CORS برای اجازه دادن به تمام مبداها
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Singleton for BADAS Service ---
badas_service: BadasOpenService | None = None

def get_badas_service() -> BadasOpenService:
    """
    EN: Lazily initialize and return a global BadasOpenService instance.
    FA: یک نمونه سراسری از BadasOpenService را به صورت lazy ساخته و برمی‌گرداند.
    """
    global badas_service
    if badas_service is None:
        logging.info("EN: Initializing BADAS-Open service for the first time... / FA: در حال مقداردهی اولیه سرویس BADAS-Open برای اولین بار...")
        badas_service = BadasOpenService(device=settings.badas_device)
        logging.info("EN: BADAS-Open service initialized. / FA: سرویس BADAS-Open مقداردهی اولیه شد.")
    return badas_service

# --- API Endpoints ---

@app.get("/status", response_model=StatusResponse)
def get_status():
    """
    EN: Health-check endpoint to verify API status and model loading.
    FA: اندپوینت بررسی سلامت برای تایید وضعیت API و بارگذاری مدل.
    """
    model_is_loaded = badas_service is not None
    return {
        "status": "ok",
        "message_en": "DriveShield API is running.",
        "message_fa": "سرویس DriveShield با موفقیت در حال اجرا است.",
        "model_loaded": model_is_loaded,
    }

@app.post("/predict/badas", response_model=BadasFullPrediction)
async def predict_badas_full(
    video: UploadFile = File(...),
    service: BadasOpenService = Depends(get_badas_service)
):
    """
    EN: Accepts a video upload, runs BADAS-Open prediction, and returns the full result.
    FA: یک ویدیوی آپلود شده را پذیرفته، پیش‌بینی BADAS-Open را اجرا کرده و نتیجه کامل را برمی‌گرداند.
    """
    logging.info(f"EN: Received video for full prediction: {video.filename} / FA: ویدیو برای پیش‌بینی کامل دریافت شد: {video.filename}")
    
    temp_dir = Path(settings.temp_dir)
    temp_video_path = None
    try:
        temp_video_path = save_upload_to_temp(video, temp_dir)
        
        # EN: Run prediction
        # FA: اجرای پیش‌بینی
        prediction_result = service.predict_video(temp_video_path.as_posix())

        return BadasFullPrediction(
            video_filename=video.filename,
            num_frames=prediction_result["num_frames"],
            max_probability=prediction_result["max_probability"],
            threshold=prediction_result["threshold"],
            num_high_risk_frames=len(prediction_result["high_risk_indices"]),
            high_risk_indices_sample=prediction_result["high_risk_indices"][:10], # EN: Sample of first 10 / FA: نمونه ۱۰ تای اول
            probabilities=prediction_result["probabilities"],
            message_en="BADAS-Open prediction completed successfully.",
            message_fa="پیش‌بینی BADAS-Open با موفقیت انجام شد.",
        )
    except Exception as e:
        logging.error(f"EN: Error during prediction: {e} / FA: خطا در هنگام پیش‌بینی: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")
    finally:
        # EN: Clean up the temporary file
        # FA: پاک‌سازی فایل موقت
        if temp_video_path:
            safe_remove_file(temp_video_path)

@app.post("/predict/badas/summary", response_model=BadasPredictionSummary)
async def predict_badas_summary(
    video: UploadFile = File(...),
    service: BadasOpenService = Depends(get_badas_service)
):
    """
    EN: Accepts a video upload, runs prediction, and returns only a summary.
    FA: یک ویدیوی آپلود شده را پذیرفته، پیش‌بینی را اجرا کرده و فقط یک خلاصه را برمی‌گرداند.
    """
    logging.info(f"EN: Received video for summary prediction: {video.filename} / FA: ویدیو برای پیش‌بینی خلاصه دریافت شد: {video.filename}")
    
    # EN: This endpoint can reuse the full prediction logic and just select fields
    # FA: این اندپوینت می‌تواند از منطق پیش‌بینی کامل استفاده کرده و فقط فیلدهای مورد نظر را انتخاب کند
    full_prediction = await predict_badas_full(video, service)
    return BadasPredictionSummary(**full_prediction.model_dump())

if __name__ == "__main__":
    """
    EN: Run the DriveShield API with Uvicorn for local development.
    FA: اجرای سرویس DriveShield با Uvicorn برای توسعه محلی.
    """
    import uvicorn
    # EN: Example to run the API locally:
    # FA: مثال دستور اجرای محلی API:
    # uvicorn backend.api.main:app --host 0.0.0.0 --port 8002 --reload
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
