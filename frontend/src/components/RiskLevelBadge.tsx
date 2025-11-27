import React from 'react';
import { useI18n } from '../i18n/I18nContext';

interface RiskLevelBadgeProps {
    maxProbability: number;
}

const RiskLevelBadge: React.FC<RiskLevelBadgeProps> = ({ maxProbability }) => {
    const { t } = useI18n();

    let text = t("riskSafe");
    let className = "inline-flex items-center rounded-full px-3 py-1 text-xs font-medium bg-emerald-500/15 text-emerald-300 border border-emerald-500/30";

    if (maxProbability >= 0.7) {
        text = t("riskHigh");
        className = "inline-flex items-center rounded-full px-3 py-1 text-xs font-medium bg-rose-500/15 text-rose-300 border border-rose-500/30";
    } else if (maxProbability >= 0.3) {
        text = t("riskMedium");
        className = "inline-flex items-center rounded-full px-3 py-1 text-xs font-medium bg-amber-500/15 text-amber-300 border border-amber-500/30";
    }

    return (
        <span className={className}>
            {text}
        </span>
    );
};

export default RiskLevelBadge;
