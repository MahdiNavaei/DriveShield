import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from typing import Dict
import logging

def compute_metrics(df: pd.DataFrame, threshold: float = 0.8) -> Dict[str, float]:
    """
    EN: Compute quantitative metrics for BADAS-Open evaluation results.
    FA: محاسبه متریک‌های کمی برای نتایج ارزیابی BADAS-Open.
    
    Args:
        df (pd.DataFrame): EN: DataFrame with 'label_cls', 'max_prob', 'num_high_risk', and 'tta_label'.
                           FA: DataFrame حاوی ستون‌های 'label_cls', 'max_prob', 'num_high_risk', و 'tta_label'.
        threshold (float): EN: The probability threshold to compute precision and recall.
                           FA: آستانه احتمال برای محاسبه دقت و بازیابی.
                           
    Returns:
        Dict[str, float]: EN: A dictionary of computed metrics. / FA: یک دیکشنری از متریک‌های محاسبه‌شده.
    """
    # EN: Drop rows with NaN in 'max_prob' to avoid errors
    # FA: حذف ردیف‌های دارای NaN در ستون 'max_prob' برای جلوگیری از خطا
    df = df.dropna(subset=['max_prob'])
    
    if df.empty:
        logging.warning("EN: DataFrame is empty after dropping NaNs. Cannot compute metrics. / FA: دیتافریم پس از حذف NaNها خالی است. امکان محاسبه متریک‌ها وجود ندارد.")
        return {}
        
    y_true = df['label_cls']
    y_score = df['max_prob']
    
    # EN: Create binary predictions based on the threshold
    # FA: ساخت پیش‌بینی‌های باینری بر اساس آستانه
    y_pred = (y_score >= threshold).astype(int)
    
    metrics = {}
    
    try:
        # EN: AUC-ROC Score
        # FA: امتیاز AUC-ROC
        metrics['auc_roc'] = roc_auc_score(y_true, y_score)
        
        # EN: Average Precision (AP)
        # FA: میانگین دقت (AP)
        metrics['average_precision'] = average_precision_score(y_true, y_score)
        
        # EN: Precision and Recall at the given threshold
        # FA: دقت و بازیابی در آستانه مشخص‌شده
        metrics[f'precision_at_{threshold}'] = precision_score(y_true, y_pred, zero_division=0)
        metrics[f'recall_at_{threshold}'] = recall_score(y_true, y_pred, zero_division=0)
        
        # EN: Approximate Mean Time-To-Accident (mTTA)
        # FA: محاسبه تقریبی میانگین زمان تا تصادف (mTTA)
        # EN: This is a rough approximation. We consider positive samples where the model fired (num_high_risk > 0).
        # FA: این یک تقریب خام است. نمونه‌های مثبتی را در نظر می‌گیریم که مدل در آنها فعال شده است (num_high_risk > 0).
        # EN: Avg frame duration is 1/30 seconds.
        # FA: میانگین زمان هر فریم ۱/۳۰ ثانیه است.
        avg_frame_duration = 1.0 / 30.0
        
        # EN: Filter for true positive samples where the model detected a risk
        # FA: فیلتر کردن نمونه‌های مثبت واقعی که مدل در آنها ریسک تشخیص داده است
        true_positives_df = df[(df['label_cls'] == 1) & (df['num_high_risk'] > 0)]
        
        if not true_positives_df.empty:
            # EN: A simple mTTA: average number of high risk frames before event * frame duration
            # FA: یک mTTA ساده: میانگین تعداد فریم‌های پرخطر قبل از حادثه * زمان هر فریم
            # EN: This assumes `tta_label` is the time of event.
            # FA: این فرض می‌کند `tta_label` زمان وقوع حادثه است.
            # EN: A better approximation would require frame-level analysis, but this is a start.
            # FA: یک تقریب بهتر نیازمند تحلیل سطح فریم است، اما این یک شروع خوب است.
            
            # EN: Let's define mTTA as the average of the ground-truth TTA for samples correctly identified as positive.
            # FA: بیایید mTTA را به عنوان میانگین TTA واقعی برای نمونه‌هایی که به درستی مثبت تشخیص داده شده‌اند، تعریف کنیم.
            mTTA = true_positives_df['tta_label'].mean()
            metrics['mTTA_seconds'] = mTTA
        else:
            metrics['mTTA_seconds'] = float('nan') # EN: No true positives detected / FA: هیچ مثبت واقعی تشخیص داده نشد

    except ValueError as e:
        logging.error(f"EN: Could not compute metrics, possibly due to only one class in y_true: {e} / FA: محاسبه متریک‌ها ممکن نبود، احتمالاً به دلیل وجود تنها یک کلاس در y_true: {e}")
        metrics['auc_roc'] = float('nan')
        metrics['average_precision'] = float('nan')
        metrics[f'precision_at_{threshold}'] = float('nan')
        metrics[f'recall_at_{threshold}'] = float('nan')
        metrics['mTTA_seconds'] = float('nan')

    logging.info("EN: Metrics computed successfully. / FA: متریک‌ها با موفقیت محاسبه شدند.")
    return metrics
