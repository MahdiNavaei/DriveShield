import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# EN: Ensure the backend modules can be imported
# FA: اطمینان از اینکه ماژول‌های backend قابل ایمپورت هستند
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.dataset.nexar_loader import create_loader
from backend.models.cnn_lstm import CnnLstmCollisionNet

# EN: Setup basic logging
# FA: راه‌اندازی لاگینگ پایه
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_args():
    """
    EN: Parse command-line arguments for the training script.
    FA: پارس کردن آرگومان‌های خط فرمان برای اسکریپت آموزش.
    """
    parser = argparse.ArgumentParser(description="EN: Train a CNN+LSTM model for collision prediction. / FA: آموزش مدل CNN+LSTM برای پیش‌بینی تصادف.")
    
    # EN: Get the project root dynamically
    # FA: گرفتن مسیر ریشه پروژه به صورت دینامیک
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    default_data_path = os.path.join(project_root, 'nexar_collision_prediction')

    parser.add_argument('--data_path', type=str, default=default_data_path,
                        help='EN: Path to the Nexar dataset directory. / FA: مسیر پوشه دیتاست Nexar.')
    parser.add_argument('--epochs', type=int, default=5,
                        help='EN: Number of training epochs. / FA: تعداد ایپوک‌های آموزش.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='EN: Batch size for training. / FA: اندازه بچ برای آموزش.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='EN: Learning rate for the optimizer. / FA: نرخ یادگیری برای بهینه‌ساز.')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help='EN: Device to use for training (cuda/cpu). / FA: دستگاه مورد استفاده برای آموزش (cuda/cpu).')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints",
                        help='EN: Directory to save model checkpoints. / FA: پوشه برای ذخیره چک‌پوینت‌های مدل.')

    return parser.parse_args()

def train_one_epoch(model, loader, optimizer, loss_cls_fn, loss_tta_fn, device, epoch, alpha):
    """
    EN: Train the model for one epoch.
    FA: آموزش مدل برای یک ایپوک.
    """
    model.train()
    total_loss, total_cls_loss, total_tta_loss = 0.0, 0.0, 0.0
    
    for i, (frames, labels) in enumerate(loader):
        # EN: Move data to the selected device
        # FA: انتقال داده‌ها به دستگاه انتخاب شده
        frames = frames.to(device)
        labels_cls = labels["cls"].to(device).float()
        labels_tta = labels["tta"].to(device)

        # EN: Forward pass
        # FA: عبور رو به جلو
        outputs = model(frames)
        logit_cls = outputs["logit_cls"]
        pred_tta = outputs["tta"]

        # EN: Compute losses
        # FA: محاسبه زیان‌ها
        cls_loss = loss_cls_fn(logit_cls, labels_cls)
        tta_loss = loss_tta_fn(pred_tta, labels_tta)
        loss = cls_loss + alpha * tta_loss

        # EN: Backward pass and optimization
        # FA: عبور رو به عقب و بهینه‌سازی
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_tta_loss += tta_loss.item()

        # EN: Log progress every 20 steps
        # FA: لاگ کردن پیشرفت هر 20 مرحله
        if (i + 1) % 20 == 0:
            logging.info(f"Epoch [{epoch+1}], Step [{i+1}/{len(loader)}], "
                         f"Loss: {loss.item():.4f}, Cls Loss: {cls_loss.item():.4f}, TTA Loss: {tta_loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    avg_cls_loss = total_cls_loss / len(loader)
    avg_tta_loss = total_tta_loss / len(loader)
    
    return avg_loss, avg_cls_loss, avg_tta_loss

def main():
    """
    EN: The main function to run the training pipeline.
    FA: تابع اصلی برای اجرای پایپ‌لاین آموزش.
    """
    args = get_args()
    logging.info("EN: Starting training with arguments: / FA: شروع آموزش با آرگومان‌های:")
    logging.info(args)

    # EN: Set the device for training
    # FA: تنظیم دستگاه برای آموزش
    device = torch.device(args.device)
    logging.info(f"EN: Using device: {device} / FA: استفاده از دستگاه: {device}")

    # EN: Create the training data loader
    # FA: ساخت DataLoader آموزشی
    logging.info("EN: Creating data loader... / FA: در حال ساخت DataLoader...")
    train_loader = create_loader(
        split="train",
        batch_size=args.batch_size,
        data_path=args.data_path,
        num_workers=4
    )
    logging.info("EN: Data loader created successfully. / FA: دیتا لودر با موفقیت ساخته شد.")

    # EN: Instantiate the model and move it to the selected device
    # FA: ساخت نمونه از مدل و انتقال آن به دستگاه انتخاب شده
    logging.info("EN: Initializing model... / FA: در حال مقداردهی اولیه مدل...")
    model = CnnLstmCollisionNet().to(device)
    logging.info("EN: Model initialized successfully. / FA: مدل با موفقیت مقداردهی اولیه شد.")

    # EN: Define the optimizer and loss functions
    # FA: تعریف بهینه‌ساز و توابع زیان
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_cls = nn.BCEWithLogitsLoss()
    loss_tta = nn.SmoothL1Loss()
    
    # EN: Alpha for weighting the TTA loss
    # FA: ضریب آلفا برای وزن‌دهی به زیان TTA
    alpha = 0.1

    # EN: Create checkpoint directory if it doesn't exist
    # FA: ساخت پوشه چک‌پوینت اگر وجود نداشته باشد
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # EN: Start the training loop
    # FA: شروع حلقه آموزش
    for epoch in range(args.epochs):
        logging.info(f"--- Starting Epoch {epoch+1}/{args.epochs} ---")
        
        avg_loss, avg_cls_loss, avg_tta_loss = train_one_epoch(
            model, train_loader, optimizer, loss_cls, loss_tta, device, epoch, alpha
        )
        
        logging.info(f"--- Epoch {epoch+1} Summary ---")
        logging.info(f"Average Loss: {avg_loss:.4f}, Avg Cls Loss: {avg_cls_loss:.4f}, Avg TTA Loss: {avg_tta_loss:.4f}")
        
        # EN: Save a checkpoint after each epoch
        # FA: ذخیره یک چک‌پوینت پس از هر ایپوک
        checkpoint_path = os.path.join(args.checkpoint_dir, f"cnn_lstm_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        logging.info(f"EN: Checkpoint saved to {checkpoint_path} / FA: چک‌پوینت در {checkpoint_path} ذخیره شد")

if __name__ == "__main__":
    main()

# EN: Example command to run training:
# FA: مثال دستور اجرا کردن آموزش:
# python -m backend.training.train_cnn_lstm --epochs 2 --batch_size 4
