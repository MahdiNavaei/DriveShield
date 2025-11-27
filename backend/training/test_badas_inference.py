import os
import sys
import argparse
import logging

# EN: Adjust sys.path to allow importing from the 'backend' package
# FA: تنظیم sys.path برای امکان وارد کردن ماژول‌ها از پکیج 'backend'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.models.badas_wrapper import BadasOpenService

# EN: Configure basic logging
# FA: پیکربندی لاگینگ پایه
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_args():
    """
    EN: Parse command-line arguments for the BADAS inference test script.
    FA: پارس کردن آرگومان‌های خط فرمان برای اسکریپت تست اینفرانس BADAS.
    """
    parser = argparse.ArgumentParser(description="EN: Test local BADAS-Open inference. / FA: تست اینفرانس محلی BADAS-Open.")
    
    parser.add_argument('--video_path', type=str, required=True,
                        help='EN: Path to an input dashcam video file (e.g., MP4). / FA: مسیر یک فایل ویدیویی داش‌کم (مثلاً MP4).')
    parser.add_argument('--device', type=str, default=None,
                        help='EN: Device to run inference on ("cuda" or "cpu"). Default is auto-detect. / FA: دستگاه برای اجرای اینفرانس ("cuda" یا "cpu"). پیش‌فرض تشخیص خودکار است.')
    
    return parser.parse_args()

def main():
    """
    EN: Entry point for testing local BADAS-Open inference on a single video file.
    FA: نقطه ورود برای تست اینفرانس محلی BADAS-Open روی یک فایل ویدیویی.
    """
    args = get_args()
    
    logging.info("EN: Starting BADAS-Open local inference test / FA: شروع تست اینفرانس محلی BADAS-Open")
    logging.info(f"EN: Video path: {args.video_path} / FA: مسیر ویدیو: {args.video_path}")
    logging.info(f"EN: Device: {'Auto-detect' if args.device is None else args.device} / FA: دستگاه: {'تشخیص خودکار' if args.device is None else args.device}")

    # EN: Check if the video file exists
    # FA: بررسی وجود فایل ویدیویی
    if not os.path.isfile(args.video_path):
        logging.error(f"EN: Video file not found at: {args.video_path} / FA: فایل ویدیویی در مسیر زیر یافت نشد: {args.video_path}")
        return

    try:
        # EN: Instantiate the BADAS service
        # FA: ساخت نمونه از سرویس BADAS
        badas_service = BadasOpenService(device=args.device)
        
        # EN: Run prediction
        # FA: اجرای پیش‌بینی
        results = badas_service.predict_video(args.video_path)
        
        # EN: Log the results
        # FA: لاگ کردن نتایج
        logging.info("--- Inference Results ---")
        logging.info(f"EN: Number of frames: {results['num_frames']} / FA: تعداد فریم‌ها: {results['num_frames']}")
        logging.info(f"EN: Max probability: {results['max_probability']:.4f} / FA: بیشترین پروبابلتی: {results['max_probability']:.4f}")
        
        num_high_risk = len(results['high_risk_indices'])
        logging.info(f"EN: Number of high-risk frames (>= {results['threshold']}): {num_high_risk} / FA: تعداد فریم‌های پرخطر (>= {results['threshold']}): {num_high_risk}")
        
        if num_high_risk > 0:
            logging.info(f"EN: First 5 high-risk frame indices: {results['high_risk_indices'][:5]} / FA: پنج ایندکس اول فریم‌های پرخطر: {results['high_risk_indices'][:5]}")
        
        logging.info("-------------------------")
        logging.info("EN: Test completed successfully! / FA: تست با موفقیت کامل شد!")

    except Exception as e:
        logging.error(f"EN: An error occurred during inference: {e} / FA: خطایی در حین اینفرانس رخ داد: {e}", exc_info=True)

if __name__ == "__main__":
    main()

# EN: Example command to test local BADAS-Open inference:
# FA: مثال دستور برای تست اینفرانس محلی BADAS-Open:
# python -m backend.training.test_badas_inference --video_path path\\to\\sample_video.mp4
