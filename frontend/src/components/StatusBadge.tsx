import React from 'react';
import { useI18n } from '../i18n/I18nContext';

interface StatusBadgeProps {
    isOnline: boolean;
    modelLoaded: boolean;
}

const StatusBadge: React.FC<StatusBadgeProps> = ({ isOnline, modelLoaded }) => {
    const { t } = useI18n();

    let text = t("statusOffline");
    let className = "inline-flex items-center rounded-full px-3 py-1 text-xs font-medium bg-rose-500/15 text-rose-300 border border-rose-500/30";

    if (isOnline) {
        if (modelLoaded) {
            text = t("statusOnlineReady");
            className = "inline-flex items-center rounded-full px-3 py-1 text-xs font-medium bg-emerald-500/15 text-emerald-300 border border-emerald-500/30";
        } else {
            text = t("statusOnlineInit");
            className = "inline-flex items-center rounded-full px-3 py-1 text-xs font-medium bg-amber-500/15 text-amber-300 border border-amber-500/30";
        }
    }

    return (
        <div className={className}>
            {text}
        </div>
    );
};

export default StatusBadge;
