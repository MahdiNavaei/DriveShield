import React, { useState, useEffect } from 'react';
import { getStatus, postBadasFull } from '../api/client';
import type { BadasFullPrediction } from '../api/client';
import { useI18n } from '../i18n/I18nContext';
import StatusBadge from '../components/StatusBadge';
import VideoUploadCard from '../components/VideoUploadCard';
import PredictionSummaryCard from '../components/PredictionSummaryCard';
import RiskTimelineChart from '../components/RiskTimelineChart';

const Dashboard: React.FC = () => {
    const { t } = useI18n();
    const [status, setStatus] = useState({ isLoading: true, isOnline: false, modelLoaded: false });
    const [prediction, setPrediction] = useState<{
        isPredicting: boolean;
        fullPrediction: BadasFullPrediction | null;
        error: string | null;
    }>({ isPredicting: false, fullPrediction: null, error: null });

    useEffect(() => {
        const checkStatus = async () => {
            try {
                const statusData = await getStatus();
                setStatus({ isLoading: false, isOnline: true, modelLoaded: statusData.model_loaded });
            } catch (error) {
                setStatus({ isLoading: false, isOnline: false, modelLoaded: false });
            }
        };
        checkStatus();
    }, []);

    const handleVideoSubmit = async (file: File) => {
        setPrediction({ isPredicting: true, fullPrediction: null, error: null });
        try {
            const predictionData = await postBadasFull(file);
            setPrediction({ isPredicting: false, fullPrediction: predictionData, error: null });
        } catch (error) {
            setPrediction({ isPredicting: false, fullPrediction: null, error: (error as Error).message });
        }
    };

    return (
        <div className="space-y-6">
            <div className="flex justify-center">
                {status.isLoading ? (
                    <p className="text-slate-400">{t("checkingBackend")}</p>
                ) : (
                    <StatusBadge isOnline={status.isOnline} modelLoaded={status.modelLoaded} />
                )}
            </div>

            <VideoUploadCard
                onSubmit={handleVideoSubmit}
                isLoading={prediction.isPredicting}
                lastError={prediction.error}
            />

            {prediction.fullPrediction && (
                <div className="space-y-6">
                    <PredictionSummaryCard summary={prediction.fullPrediction} />
                    <RiskTimelineChart 
                        probabilities={prediction.fullPrediction.probabilities} 
                        threshold={prediction.fullPrediction.threshold} 
                    />
                </div>
            )}
        </div>
    );
};

export default Dashboard;
