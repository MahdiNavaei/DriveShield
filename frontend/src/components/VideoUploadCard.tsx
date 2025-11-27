import React, { useState } from 'react';
import { useI18n } from '../i18n/I18nContext';

interface VideoUploadCardProps {
    onSubmit: (file: File) => Promise<void>;
    isLoading: boolean;
    lastError: string | null;
}

const VideoUploadCard: React.FC<VideoUploadCardProps> = ({ onSubmit, isLoading, lastError }) => {
    const { t } = useI18n();
    const [selectedFile, setSelectedFile] = useState<File | null>(null);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        if (event.target.files && event.target.files[0]) {
            setSelectedFile(event.target.files[0]);
        }
    };

    const handleSubmit = (event: React.FormEvent) => {
        event.preventDefault();
        if (selectedFile) {
            onSubmit(selectedFile);
        }
    };

    return (
        <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
            <h2 className="text-xl font-semibold mb-4 text-center text-slate-100">
                {t("uploadTitle")}
            </h2>
            <form onSubmit={handleSubmit} className="space-y-4">
                <input
                    type="file"
                    accept="video/*"
                    onChange={handleFileChange}
                    className="block w-full text-sm text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-slate-800 file:text-slate-200 hover:file:bg-slate-700 cursor-pointer"
                />
                <button
                    type="submit"
                    disabled={!selectedFile || isLoading}
                    className="w-full bg-blue-600 text-white py-3 px-4 rounded-lg hover:bg-blue-700 disabled:bg-slate-700 disabled:text-slate-400 disabled:cursor-not-allowed transition-colors font-medium"
                >
                    {isLoading ? t("uploadingLabel") : t("uploadButton")}
                </button>
            </form>
            {lastError && (
                <div className="mt-4 p-3 bg-rose-500/15 border border-rose-500/30 rounded-lg">
                    <p className="text-rose-300 text-sm text-center">
                        {t("errorLabel")}: {lastError}
                    </p>
                </div>
            )}
        </div>
    );
};

export default VideoUploadCard;
