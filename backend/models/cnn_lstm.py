import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from typing import Dict

class CnnLstmCollisionNet(nn.Module):
    """
    EN: A CNN+LSTM model for collision prediction from video frames.
        It uses a pretrained ResNet-18 to extract features from each frame,
        followed by an LSTM to model temporal dependencies.
    FA: یک مدل CNN+LSTM برای پیش‌بینی تصادف از فریم‌های ویدئویی.
        این مدل از یک ResNet-18 از پیش آموزش‌دیده برای استخراج ویژگی از هر فریم
        و سپس یک LSTM برای مدل‌سازی وابستگی‌های زمانی استفاده می‌کند.
    """
    def __init__(self, num_frames: int = 60, hidden_size: int = 256, feature_size: int = 512):
        """
        EN: Initializes the model layers.
        FA: لایه‌های مدل را مقداردهی اولیه می‌کند.
        
        Args:
            num_frames (int): EN: Number of frames in each input window. / FA: تعداد فریم در هر پنجره ورودی.
            hidden_size (int): EN: Hidden size of the LSTM layer. / FA: اندازه پنهان لایه LSTM.
            feature_size (int): EN: Feature vector size from the CNN backbone. / FA: اندازه بردار ویژگی از backbone سی‌ان‌ان.
        """
        super().__init__()
        
        # EN: Load a pretrained ResNet-18 model as the CNN backbone
        # FA: بارگذاری مدل ResNet-18 از پیش آموزش‌دیده به عنوان backbone سی‌ان‌ان
        weights = ResNet18_Weights.DEFAULT
        self.resnet = resnet18(weights=weights)
        
        # EN: Remove the final fully connected layer to get feature vectors
        # FA: حذف لایه fully connected نهایی برای گرفتن بردارهای ویژگی
        self.resnet.fc = nn.Identity()

        # EN: Bidirectional LSTM to process the sequence of frame features
        # FA: یک LSTM دوطرفه برای پردازش توالی ویژگی‌های فریم
        self.lstm = nn.LSTM(
            input_size=feature_size, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )

        # EN: The output size from the bidirectional LSTM is 2 * hidden_size
        # FA: اندازه خروجی از LSTM دوطرفه برابر است با 2 * hidden_size
        lstm_output_size = hidden_size * 2

        # EN: Output head for collision classification (predicts a logit)
        # FA: هد خروجی برای طبقه‌بندی تصادف (یک لاجیت پیش‌بینی می‌کند)
        self.head_cls = nn.Linear(lstm_output_size, 1)

        # EN: Output head for Time-To-Accident (TTA) regression
        # FA: هد خروجی برای رگرسیون زمان تا تصادف (TTA)
        self.head_tta = nn.Linear(lstm_output_size, 1)

    def forward(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        EN: Defines the forward pass of the model.
        FA: عبور رو به جلوی مدل را تعریف می‌کند.

        Args:
            frames (torch.Tensor): EN: Input tensor of shape (B, T, C, H, W). / FA: تنسور ورودی با شکل (B, T, C, H, W).

        Returns:
            Dict[str, torch.Tensor]: EN: A dictionary with 'logit_cls' and 'tta' tensors. / FA: یک دیکشنری با تنسورهای 'logit_cls' و 'tta'.
        """
        # EN: Get dimensions
        # FA: گرفتن ابعاد
        batch_size, num_frames, C, H, W = frames.shape

        # EN: Reshape the input to process all frames at once with the CNN
        # FA: تغییر شکل ورودی برای پردازش تمام فریم‌ها به یکباره با CNN
        frames_reshaped = frames.view(batch_size * num_frames, C, H, W)
        
        # EN: Extract features for each frame
        # FA: استخراج ویژگی برای هر فریم
        with torch.no_grad(): # EN: Freeze ResNet weights / FA: فریز کردن وزن‌های ResNet
            features = self.resnet(frames_reshaped)
        
        # EN: Reshape features back to (B, T, D) for the LSTM
        # FA: تغییر شکل ویژگی‌ها به (B, T, D) برای LSTM
        features_seq = features.view(batch_size, num_frames, -1)

        # EN: Pass the sequence of features through the LSTM
        # FA: عبور دادن توالی ویژگی‌ها از LSTM
        lstm_out, _ = self.lstm(features_seq)
        
        # EN: Apply temporal max pooling over the LSTM outputs
        # FA: اعمال max pooling زمانی روی خروجی‌های LSTM
        pooled_out = torch.max(lstm_out, dim=1)[0]
        
        # EN: Get the final predictions from the output heads
        # FA: گرفتن پیش‌بینی‌های نهایی از هدهای خروجی
        logit_cls = self.head_cls(pooled_out).squeeze(-1)
        tta = self.head_tta(pooled_out).squeeze(-1)
        
        return {
            "logit_cls": logit_cls,
            "tta": tta
        }
