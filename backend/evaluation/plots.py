import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from typing import Dict
import os
import logging

def plot_roc_curve(df: pd.DataFrame, metrics: Dict[str, float], output_dir: str):
    """
    EN: Plots and saves the ROC curve.
    FA: نمودار ROC را رسم و ذخیره می‌کند.
    """
    # EN: Drop rows with NaN in 'max_prob' to avoid errors
    # FA: حذف ردیف‌های دارای NaN در ستون 'max_prob' برای جلوگیری از خطا
    df = df.dropna(subset=['max_prob'])
    
    if df.empty or df['label_cls'].nunique() < 2:
        logging.warning("EN: Cannot plot ROC curve. DataFrame is empty or contains only one class after dropping NaNs. / FA: امکان رسم نمودار ROC وجود ندارد. دیتافریم خالی است یا پس از حذف NaNها فقط یک کلاس دارد.")
        return

    y_true = df['label_cls']
    y_score = df['max_prob']
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = metrics.get('auc_roc', auc(fpr, tpr))
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate / نرخ مثبت کاذب')
    plt.ylabel('True Positive Rate / نرخ مثبت واقعی')
    plt.title('Receiver Operating Characteristic (ROC) Curve\nنمودار مشخصه عملکرد گیرنده (ROC)')
    plt.legend(loc="lower right")
    
    output_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(output_path)
    plt.close()

def plot_precision_recall_curve(df: pd.DataFrame, metrics: Dict[str, float], output_dir: str):
    """
    EN: Plots and saves the Precision-Recall curve.
    FA: نمودار دقت-بازیابی را رسم و ذخیره می‌کند.
    """
    # EN: Drop rows with NaN in 'max_prob' to avoid errors
    # FA: حذف ردیف‌های دارای NaN در ستون 'max_prob' برای جلوگیری از خطا
    df = df.dropna(subset=['max_prob'])

    if df.empty or df['label_cls'].nunique() < 2:
        logging.warning("EN: Cannot plot Precision-Recall curve. DataFrame is empty or contains only one class after dropping NaNs. / FA: امکان رسم نمودار دقت-بازیابی وجود ندارد. دیتافریم خالی است یا پس از حذف NaNها فقط یک کلاس دارد.")
        return

    y_true = df['label_cls']
    y_score = df['max_prob']
    
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = metrics.get('average_precision', auc(recall, precision))

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {ap:.2f})')
    plt.xlabel('Recall / بازیابی')
    plt.ylabel('Precision / دقت')
    plt.title('Precision-Recall Curve\nنمودار دقت-بازیابی')
    plt.legend(loc="upper right")
    plt.ylim([0.0, 1.05])
    
    output_path = os.path.join(output_dir, 'pr_curve.png')
    plt.savefig(output_path)
    plt.close()
