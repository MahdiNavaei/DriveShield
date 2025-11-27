import os
import sys
import argparse
import logging
import json
import pandas as pd

# EN: Adjust sys.path to allow importing from the 'backend' package
# FA: تنظیم sys.path برای امکان وارد کردن ماژول‌ها از پکیج 'backend'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.models.badas_wrapper import BadasOpenService
from backend.evaluation.eval_dataset import BadasEvaluationRunner
from backend.evaluation.metrics import compute_metrics
from backend.evaluation.plots import plot_roc_curve, plot_precision_recall_curve

# EN: Configure basic logging
# FA: پیکربندی لاگینگ پایه
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_args():
    """
    EN: Parse command-line arguments for the BADAS evaluation script.
    FA: پارس کردن آرگومان‌های خط فرمان برای اسکریپت ارزیابی BADAS.
    """
    parser = argparse.ArgumentParser(description="EN: Run end-to-end evaluation for BADAS-Open. / FA: اجرای ارزیابی سرتاسری برای BADAS-Open.")
    
    parser.add_argument('--split', type=str, default='test_public', choices=['train', 'test_public', 'test_private'],
                        help='EN: Dataset split to evaluate on. / FA: اسپلیت دیتاست برای ارزیابی.')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='EN: Number of unique video samples to process. / FA: تعداد نمونه‌های ویدیویی یکتا برای پردازش.')
    parser.add_argument('--output_dir', type=str, default='eval_output',
                        help='EN: Directory to save evaluation results (CSV, JSON, plots). / FA: پوشه برای ذخیره نتایج ارزیابی.')
    parser.add_argument('--device', type=str, default=None,
                        help='EN: Device to run inference on ("cuda" or "cpu"). Default is auto-detect. / FA: دستگاه برای اجرای اینفرانس.')

    return parser.parse_args()

def main():
    """
    EN: Main entry point for the BADAS-Open evaluation pipeline.
    FA: نقطه ورود اصلی برای پایپ‌لاین ارزیابی BADAS-Open.
    """
    args = get_args()
    logging.info("EN: Starting BADAS-Open evaluation pipeline... / FA: شروع پایپ‌لاین ارزیابی BADAS-Open...")
    logging.info(f"Args: {args}")

    # EN: Create output directory if it doesn't exist
    # FA: ساخت پوشه خروجی در صورت عدم وجود
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # 1. EN: Instantiate the BADAS service
        #    FA: ساخت نمونه از سرویس BADAS
        service = BadasOpenService(device=args.device)

        # 2. EN: Initialize and run the evaluation runner
        #    FA: مقداردهی اولیه و اجرای رانر ارزیابی
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        data_path = os.path.join(project_root, 'nexar_collision_prediction')
        
        runner = BadasEvaluationRunner(service, args.split, args.num_samples, data_path)
        results_df = runner.run()

        if results_df.empty:
            logging.warning("EN: No results were generated. Exiting. / FA: هیچ نتیجه‌ای تولید نشد. خروج.")
            return

        # 3. EN: Save raw results to CSV
        #    FA: ذخیره نتایج خام در فایل CSV
        csv_path = os.path.join(args.output_dir, 'results.csv')
        results_df.to_csv(csv_path, index=False)
        logging.info(f"EN: Raw results saved to {csv_path} / FA: نتایج خام در {csv_path} ذخیره شد.")

        # 4. EN: Compute metrics
        #    FA: محاسبه متریک‌ها
        metrics = compute_metrics(results_df)
        
        # 5. EN: Save metrics to JSON
        #    FA: ذخیره متریک‌ها در فایل JSON
        metrics_path = os.path.join(args.output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"EN: Metrics saved to {metrics_path} / FA: متریک‌ها در {metrics_path} ذخیره شد.")

        # 6. EN: Generate and save plots
        #    FA: تولید و ذخیره نمودارها
        plot_roc_curve(results_df, metrics, args.output_dir)
        plot_precision_recall_curve(results_df, metrics, args.output_dir)
        logging.info(f"EN: Plots saved in {args.output_dir} / FA: نمودارها در {args.output_dir} ذخیره شدند.")

        # 7. EN: Log a summary
        #    FA: لاگ کردن خلاصه
        logging.info("--- Evaluation Summary ---")
        logging.info(f"EN: Processed {len(results_df)} samples from '{args.split}' split. / FA: تعداد {len(results_df)} نمونه از اسپلیت '{args.split}' پردازش شد.")
        logging.info(f"EN: Positive samples: {results_df['label_cls'].sum()} / FA: نمونه‌های مثبت: {results_df['label_cls'].sum()}")
        logging.info(f"EN: AUC-ROC: {metrics.get('auc_roc', 'N/A'):.4f} / FA: AUC-ROC: {metrics.get('auc_roc', 'N/A'):.4f}")
        logging.info(f"EN: Average Precision (AP): {metrics.get('average_precision', 'N/A'):.4f} / FA: میانگین دقت (AP): {metrics.get('average_precision', 'N/A'):.4f}")
        logging.info(f"EN: Precision @ 0.8: {metrics.get('precision_at_0.8', 'N/A'):.4f} / FA: دقت در آستانه ۰.۸: {metrics.get('precision_at_0.8', 'N/A'):.4f}")
        logging.info(f"EN: Recall @ 0.8: {metrics.get('recall_at_0.8', 'N/A'):.4f} / FA: بازیابی در آستانه ۰.۸: {metrics.get('recall_at_0.8', 'N/A'):.4f}")
        logging.info(f"EN: mTTA (seconds): {metrics.get('mTTA_seconds', 'N/A'):.2f} / FA: میانگین زمان تا تصادف (ثانیه): {metrics.get('mTTA_seconds', 'N/A'):.2f}")
        logging.info("------------------------")

    except Exception as e:
        logging.error(f"EN: An error occurred during the evaluation pipeline: {e} / FA: خطایی در پایپ‌لاین ارزیابی رخ داد: {e}", exc_info=True)

if __name__ == "__main__":
    main()

# EN: Example command to run evaluation:
# FA: مثال دستور اجرای ارزیابی:
# python -m backend.evaluation.run_eval_badas --split test-public --num_samples 50 --output_dir eval_output
