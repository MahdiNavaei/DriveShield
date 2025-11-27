import os
import sys
import argparse
import logging
import json
from typing import Dict, List

import torch
import torch.nn.functional as F

# EN: Adjust sys.path to allow importing from the 'backend' directory
# FA: تنظیم sys.path برای امکان وارد کردن ماژول‌ها از پوشه 'backend'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.dataset.nexar_loader import create_loader
from backend.models.cnn_lstm import CnnLstmCollisionNet
from backend.utils.metrics import compute_binary_classification_metrics

# EN: Setup basic logging configuration
# FA: راه‌اندازی تنظیمات پایه برای لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(checkpoint_path: str, device: torch.device) -> CnnLstmCollisionNet:
    """
    EN: Loads a trained model from a checkpoint file.
    FA: بارگذاری یک مدل آموزش‌دیده از یک فایل چک‌پوینت.
    
    Args:
        checkpoint_path (str): EN: The path to the .pt checkpoint file. / FA: مسیر فایل چک‌پوینت .pt.
        device (torch.device): EN: The device to load the model onto. / FA: دستگاهی که مدل باید روی آن بارگذاری شود.
        
    Returns:
        CnnLstmCollisionNet: EN: The loaded model in evaluation mode. / FA: مدل بارگذاری‌شده در حالت ارزیابی.
    """
    logging.info(f"EN: Loading model from checkpoint: {checkpoint_path} / FA: در حال بارگذاری مدل از چک‌پوینت: {checkpoint_path}")
    # EN: Instantiate the model architecture
    # FA: ساخت نمونه از معماری مدل
    model = CnnLstmCollisionNet()
    
    # EN: Load the checkpoint, mapping storage to the specified device
    # FA: بارگذاری چک‌پوینت، با نگاشت حافظه به دستگاه مشخص‌شده
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # EN: Load the state dictionary into the model
    # FA: بارگذاری state dictionary در مدل
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # EN: Move the model to the specified device
    # FA: انتقال مدل به دستگاه مشخص‌شده
    model.to(device)
    
    # EN: Set the model to evaluation mode
    # FA: تنظیم مدل به حالت ارزیابی
    model.eval()
    
    logging.info("EN: Model loaded successfully. / FA: مدل با موفقیت بارگذاری شد.")
    return model

def evaluate_model(model: CnnLstmCollisionNet, loader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, float]:
    """
    EN: Evaluates the model on a given dataset split and computes metrics.
    FA: ارزیابی مدل روی یک اسپلیت مشخص از دیتاست و محاسبه متریک‌ها.
    
    Args:
        model (CnnLstmCollisionNet): EN: The model to evaluate. / FA: مدلی که باید ارزیابی شود.
        loader (torch.utils.data.DataLoader): EN: The DataLoader for the evaluation data. / FA: دیتا لودر برای داده‌های ارزیابی.
        device (torch.device): EN: The device to run evaluation on. / FA: دستگاهی که ارزیابی روی آن اجرا می‌شود.
        
    Returns:
        Dict[str, float]: EN: A dictionary containing the computed classification metrics. / FA: یک دیکشنری حاوی متریک‌های طبقه‌بندی محاسبه‌شده.
    """
    # EN: Ensure model is in evaluation mode and disable gradient calculations
    # FA: اطمینان از اینکه مدل در حالت ارزیابی است و غیرفعال کردن محاسبه گرادیان‌ها
    model.eval()
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for i, (frames, labels) in enumerate(loader):
            # EN: Move data to the selected device
            # FA: انتقال داده‌ها به دستگاه انتخاب شده
            frames = frames.to(device)
            labels_cls = labels["cls"] # Keep on CPU for easier appending

            # EN: Forward pass to get model outputs
            # FA: عبور رو به جلو برای گرفتن خروجی‌های مدل
            outputs = model(frames)
            logits = outputs["logit_cls"]
            
            # EN: Convert logits to probabilities using sigmoid
            # FA: تبدیل لاجیت‌ها به پروبابلتی با استفاده از سیگموید
            probs = torch.sigmoid(logits)
            
            # EN: Append batch results to the lists (move to CPU)
            # FA: اضافه کردن نتایج بچ به لیست‌ها (انتقال به CPU)
            all_labels.append(labels_cls.cpu())
            all_probs.append(probs.cpu())

            if (i + 1) % 20 == 0:
                logging.info(f"EN: Processing batch {i+1}/{len(loader)} / FA: در حال پردازش بچ {i+1}/{len(loader)}")

    # EN: Concatenate all batch results into single tensors
    # FA: الحاق تمام نتایج بچ‌ها به تنسورهای واحد
    all_labels_tensor = torch.cat(all_labels)
    all_probs_tensor = torch.cat(all_probs)
    
    logging.info("EN: All batches processed. Computing metrics... / FA: تمام بچ‌ها پردازش شدند. در حال محاسبه متریک‌ها...")
    
    # EN: Compute and return the metrics
    # FA: محاسبه و بازگرداندن متریک‌ها
    metrics = compute_binary_classification_metrics(all_labels_tensor, all_probs_tensor, threshold=0.5)
    return metrics

def get_args():
    """
    EN: Parse command-line arguments for the evaluation script.
    FA: پارس کردن آرگومان‌های خط فرمان برای اسکریپت ارزیابی.
    """
    parser = argparse.ArgumentParser(description="EN: Evaluate a trained CNN+LSTM model. / FA: ارزیابی یک مدل آموزش‌دیده CNN+LSTM.")
    
    # EN: Dynamically determine the default data path based on project structure
    # FA: تعیین پویای مسیر پیش‌فرض داده بر اساس ساختار پروژه
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    default_data_path = os.path.join(project_root, 'nexar_collision_prediction')

    parser.add_argument('--data_path', type=str, default=default_data_path,
                        help='EN: Path to the Nexar dataset directory. / FA: مسیر پوشه دیتاست Nexar.')
    parser.add_argument('--split', type=str, default='test_public', choices=['train', 'test_public', 'test_private'],
                        help='EN: Dataset split to evaluate on. / FA: اسپلیت دیتاست برای ارزیابی.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='EN: Batch size for evaluation. / FA: اندازه بچ برای ارزیابی.')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help='EN: Device to use for evaluation (cuda/cpu). / FA: دستگاه مورد استفاده برای ارزیابی (cuda/cpu).')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='EN: Path to a trained model checkpoint file. / FA: مسیر فایل چک‌پوینت مدل آموزش‌دیده.')
    parser.add_argument('--metrics_output', type=str, default='eval_results.json',
                        help='EN: Path to save the output metrics JSON file. / FA: مسیر برای ذخیره فایل JSON متریک‌های خروجی.')

    return parser.parse_args()

def main():
    """
    EN: Main entry point for evaluating the CNN+LSTM collision prediction model.
    FA: نقطه ورود اصلی برای ارزیابی مدل پیش‌بینی تصادف CNN+LSTM.
    """
    args = get_args()
    logging.info("EN: Starting evaluation with arguments: / FA: شروع ارزیابی با آرگومان‌های:")
    logging.info(args)

    # EN: Set the device for evaluation
    # FA: تنظیم دستگاه برای ارزیابی
    device = torch.device(args.device)
    logging.info(f"EN: Using device: {device} / FA: استفاده از دستگاه: {device}")

    # EN: Create the data loader for the specified split
    # FA: ساخت دیتا لودر برای اسپلیت مشخص‌شده
    logging.info(f"EN: Creating data loader for '{args.split}' split... / FA: در حال ساخت دیتا لودر برای اسپلیت '{args.split}'...")
    loader = create_loader(
        split=args.split,
        batch_size=args.batch_size,
        data_path=args.data_path,
        num_workers=4
    )

    # EN: Load the trained model from the checkpoint
    # FA: بارگذاری مدل آموزش‌دیده از چک‌پوینت
    model = load_model(args.checkpoint_path, device)

    # EN: Run the evaluation
    # FA: اجرای ارزیابی
    metrics = evaluate_model(model, loader, device)

    # EN: Log the computed metrics
    # FA: لاگ کردن متریک‌های محاسبه‌شده
    logging.info("--- Evaluation Metrics ---")
    for key, value in metrics.items():
        logging.info(f"EN: {key.capitalize()} = {value:.4f} / FA: {key.capitalize()} = {value:.4f}")
    logging.info("------------------------")

    # EN: Save the metrics to a JSON file
    # FA: ذخیره متریک‌ها در یک فایل JSON
    output_data = {
        "split": args.split,
        "checkpoint_path": args.checkpoint_path,
        "metrics": metrics
    }
    
    with open(args.metrics_output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logging.info(f"EN: Evaluation results saved to {args.metrics_output} / FA: نتایج ارزیابی در {args.metrics_output} ذخیره شد")

if __name__ == "__main__":
    main()

# EN: Example command to run evaluation:
# FA: مثال دستور اجرای ارزیابی:
# python -m backend.training.eval_cnn_lstm --split test_public --checkpoint_path checkpoints/cnn_lstm_epoch_1.pt
