import shutil
import uuid
import logging
from pathlib import Path
from fastapi import UploadFile

def save_upload_to_temp(upload_file: UploadFile, temp_dir: Path) -> Path:
    """
    EN: Save an uploaded video file to a temporary directory and return its path.
    FA: فایل ویدیوی آپلود شده را در یک دایرکتوری موقت ذخیره کرده و مسیر آن را برمی‌گرداند.
    """
    # EN: Ensure the temporary directory exists
    # FA: اطمینان از وجود دایرکتوری موقت
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # EN: Create a unique filename to avoid collisions
    # FA: ایجاد یک نام فایل یکتا برای جلوگیری از تداخل
    file_extension = Path(upload_file.filename).suffix
    temp_filename = f"{uuid.uuid4()}{file_extension}"
    temp_file_path = temp_dir / temp_filename
    
    # EN: Write the uploaded file to the disk in chunks
    # FA: نوشتن فایل آپلود شده روی دیسک به صورت تکه‌ای
    try:
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        logging.info(f"EN: Saved uploaded file to {temp_file_path} / FA: فایل آپلود شده در {temp_file_path} ذخیره شد.")
    finally:
        upload_file.file.close()
        
    return temp_file_path

def safe_remove_file(path: Path | str) -> None:
    """
    EN: Safely remove a file if it exists, ignoring errors.
    FA: در صورت موجود بودن، فایل را به صورت امن حذف می‌کند و خطاها را نادیده می‌گیرد.
    """
    try:
        path_obj = Path(path)
        if path_obj.exists() and path_obj.is_file():
            path_obj.unlink()
            logging.info(f"EN: Safely removed temporary file: {path} / FA: فایل موقت با موفقیت حذف شد: {path}")
    except Exception as e:
        # EN: Log the error but do not raise it, as it's a non-critical cleanup step
        # FA: خطا را لاگ کن اما آن را بالا نبر، زیرا این یک مرحله پاک‌سازی غیرحیاتی است
        logging.warning(f"EN: Could not remove file {path}: {e} / FA: حذف فایل {path} با خطا مواجه شد: {e}")
