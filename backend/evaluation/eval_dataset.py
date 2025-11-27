import os
import sys
import logging
import pandas as pd
from typing import List, Dict, Any

# EN: Adjust sys.path to allow importing from the 'backend' package
# FA: تنظیم sys.path برای امکان وارد کردن ماژول‌ها از پکیج 'backend'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.dataset.nexar_loader import load_nexar_local # Changed import
from backend.models.badas_wrapper import BadasOpenService
from tqdm import tqdm

class BadasEvaluationRunner:
    """
    EN: Run BADAS-Open on Nexar dataset windows and collect predictions.
    FA: اجرای BADAS-Open روی پنجره‌های دیتاست Nexar و جمع‌آوری پیش‌بینی‌ها.
    """
    def __init__(self, service: BadasOpenService, split: str, num_samples: int, data_path: str):
        """
        EN: Initializes the evaluation runner.
        FA: راه‌اندازی اجراکننده ارزیابی.
        
        Args:
            service (BadasOpenService): EN: The BADAS service instance. / FA: نمونه سرویس BADAS.
            split (str): EN: The dataset split to use ('train', 'test_public', etc.). / FA: اسپلیت دیتاست مورد استفاده.
            num_samples (int): EN: The number of unique video samples to process. / FA: تعداد نمونه‌های ویدیویی یکتا برای پردازش.
            data_path (str): EN: Path to the root of the Nexar dataset. / FA: مسیر ریشه دیتاست Nexar.
        """
        self.service = service
        self.split = split
        self.num_samples = num_samples
        self.data_path = data_path
        self.results = []

    def run(self) -> pd.DataFrame:
        """
        EN: Runs the evaluation loop by iterating through raw samples, collects results, and returns them as a DataFrame.
        FA: حلقه ارزیابی را با تکرار روی نمونه‌های خام اجرا کرده، نتایج را جمع‌آوری و به صورت یک DataFrame برمی‌گرداند.
        """
        logging.info(f"EN: Starting evaluation run on '{self.split}' split for {self.num_samples} samples. / FA: شروع اجرای ارزیابی روی اسپلیت '{self.split}' برای {self.num_samples} نمونه.")
        
        # EN: Load the raw samples list to get access to video paths
        # FA: بارگذاری لیست نمونه‌های خام برای دسترسی به مسیرهای ویدیو
        all_splits = load_nexar_local(path=self.data_path)
        samples_to_process = all_splits.get(self.split.replace("-", "_"))
        
        if not samples_to_process:
            logging.error(f"EN: Split '{self.split}' not found or is empty. / FA: اسپلیت '{self.split}' یافت نشد یا خالی است.")
            return pd.DataFrame()

        # EN: Limit the number of samples to process
        # FA: محدود کردن تعداد نمونه‌ها برای پردازش
        samples_to_process = samples_to_process[:self.num_samples]
        
        self.results = []
        for sample in tqdm(samples_to_process, desc=f"EN: Evaluating {self.split} / FA: در حال ارزیابی {self.split}"):
            video_path = sample["video_path"]
            
            try:
                # EN: Run inference using the BADAS service
                # FA: اجرای اینفرانس با استفاده از سرویس BADAS
                badas_result = self.service.predict_video(video_path)
                
                # EN: Collect the ground truth and prediction
                # FA: جمع‌آوری برچسب واقعی و پیش‌بینی
                self.results.append({
                    "video_path": video_path,
                    "label_cls": sample["label"],
                    "max_prob": badas_result["max_probability"],
                    "num_high_risk": len(badas_result["high_risk_indices"]),
                    "tta_label": sample.get("time_of_event", -1) # Use time_of_event for TTA label
                })
            except Exception as e:
                logging.error(f"EN: Failed to process video {video_path}: {e} / FA: پردازش ویدیو {video_path} با خطا مواجه شد: {e}")

        logging.info(f"EN: Evaluation finished. Processed {len(self.results)} samples. / FA: ارزیابی تمام شد. {len(self.results)} نمونه پردازش شد.")
        return pd.DataFrame(self.results)

    def save_results_csv(self, output_path: str):
        """
        EN: Saves the collected evaluation results to a CSV file.
        FA: نتایج ارزیابی جمع‌آوری‌شده را در یک فایل CSV ذخیره می‌کند.
        """
        if not self.results:
            logging.warning("EN: No results to save. / FA: نتیجه‌ای برای ذخیره وجود ندارد.")
            return
            
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)
        logging.info(f"EN: Evaluation results saved to {output_path} / FA: نتایج ارزیابی در {output_path} ذخیره شد.")
