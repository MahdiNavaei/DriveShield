import React from 'react';
import type { BadasPredictionSummary } from '../api/client';
import { useI18n } from '../i18n/I18nContext';
import RiskLevelBadge from './RiskLevelBadge';

interface PredictionSummaryCardProps {
    summary: BadasPredictionSummary;
}

const PredictionSummaryCard: React.FC<PredictionSummaryCardProps> = ({ summary }) => {
    const { t } = useI18n();

    return (
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl mt-8">
            <h2 className="text-xl font-semibold mb-4 text-center text-slate-100">
                {t("summaryTitle")}
            </h2>
            <div className="space-y-3">
                <div className="flex justify-between items-center py-2 border-b border-slate-800">
                    <span className="font-semibold text-slate-300">{t("fileNameLabel")}:</span>
                    <span className="text-slate-100">{summary.video_filename}</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-slate-800">
                    <span className="font-semibold text-slate-300">{t("maxProbLabel")}:</span>
                    <span className="text-slate-100">{(summary.max_probability * 100).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-slate-800">
                    <span className="font-semibold text-slate-300">{t("thresholdLabel")}:</span>
                    <span className="text-slate-100">{summary.threshold}</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-slate-800">
                    <span className="font-semibold text-slate-300">{t("highRiskFramesLabel")}:</span>
                    <span className="text-slate-100">{summary.num_high_risk_frames}</span>
                </div>
                <div className="flex justify-between items-center py-2">
                    <span className="font-semibold text-slate-300">{t("riskLevelLabel")}:</span>
                    <RiskLevelBadge maxProbability={summary.max_probability} />
                </div>
                {summary.high_risk_indices_sample.length > 0 && (
                    <div className="pt-4 mt-4 border-t border-slate-800">
                        <h3 className="font-semibold text-slate-300 mb-2">{t("sampleIndicesLabel")}:</h3>
                        <p className="text-slate-400 text-sm break-all">
                            {summary.high_risk_indices_sample.join(', ')}
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default PredictionSummaryCard;
