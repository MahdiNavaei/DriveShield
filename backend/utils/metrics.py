import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict
import logging

def compute_binary_classification_metrics(
    y_true: torch.Tensor, 
    y_prob: torch.Tensor, 
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    EN: Compute common binary classification metrics for collision prediction.
    FA: محاسبه متریک‌های رایج طبقه‌بندی باینری برای پیش‌بینی تصادف.

    Args:
        y_true (torch.Tensor): EN: Ground truth labels (0/1), shape (N,). / FA: لیبل‌های واقعی (۰/۱)، شکل (N,).
        y_prob (torch.Tensor): EN: Predicted probabilities (between 0 and 1), shape (N,). / FA: پروبابلتی‌های پیش‌بینی‌شده (بین ۰ و ۱)، شکل (N,).
        threshold (float): EN: Cutoff for converting probabilities to binary predictions. / FA: آستانه برای تبدیل پروبابلتی‌ها به پیش‌بینی باینری.

    Returns:
        Dict[str, float]: EN: A dictionary of computed metrics. / FA: یک دیکشنری از متریک‌های محاسبه‌شده.
    """
    # EN: Convert tensors to numpy arrays
    # FA: تبدیل تنسورها به آرایه‌های نام‌پای
    y_true_np = y_true.cpu().numpy()
    y_prob_np = y_prob.cpu().numpy()

    # EN: Compute binary predictions based on the threshold
    # FA: محاسبه پیش‌بینی‌های باینری بر اساس آستانه
    y_pred_np = (y_prob_np >= threshold).astype(int)
    
    metrics = {}
    
    # EN: Calculate metrics
    # FA: محاسبه متریک‌ها
    metrics["accuracy"] = accuracy_score(y_true_np, y_pred_np)
    metrics["precision"] = precision_score(y_true_np, y_pred_np, zero_division=0)
    metrics["recall"] = recall_score(y_true_np, y_pred_np, zero_division=0)
    metrics["f1"] = f1_score(y_true_np, y_pred_np, zero_division=0)
    
    # EN: Handle potential ValueError for roc_auc_score if only one class is present
    # FA: مدیریت خطای احتمالی ValueError برای roc_auc_score در صورتی که فقط یک کلاس وجود داشته باشد
    try:
        metrics["roc_auc"] = roc_auc_score(y_true_np, y_prob_np)
    except ValueError as e:
        logging.warning(f"EN: Could not compute ROC-AUC score: {e}. Setting to NaN. / FA: امکان محاسبه امتیاز ROC-AUC وجود نداشت: {e}. مقدار NaN تنظیم شد.")
        metrics["roc_auc"] = float("nan")
        
    return metrics
