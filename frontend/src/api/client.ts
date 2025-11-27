import axios from 'axios';

// EN: Define the base URL for the backend API
// FA: تعریف URL پایه برای API بک‌اند
const apiClient = axios.create({
    baseURL: 'http://localhost:8002',
});

// EN: Define TypeScript interfaces matching the FastAPI schemas
// FA: تعریف اینترفیس‌های TypeScript مطابق با اسکیماهای FastAPI
export interface StatusResponse {
    status: string;
    message_en: string;
    message_fa: string;
    model_loaded: boolean;
}

export interface BadasPredictionSummary {
    video_filename: string;
    num_frames: number;
    max_probability: number;
    threshold: number;
    num_high_risk_frames: number;
    high_risk_indices_sample: number[];
    message_en: string;
    message_fa: string;
}

export interface BadasFullPrediction extends BadasPredictionSummary {
    probabilities: number[];
}

/**
 * EN: Call /status endpoint to check API and model status.
 * FA: فراخوانی اندپوینت /status برای بررسی وضعیت API و مدل.
 */
export const getStatus = async (): Promise<StatusResponse> => {
    try {
        const response = await apiClient.get<StatusResponse>('/status');
        return response.data;
    } catch (error) {
        console.error("Error fetching API status:", error);
        throw new Error("Failed to connect to the backend API.");
    }
};

/**
 * EN: Upload a video file and get a full BADAS-Open prediction.
 * FA: آپلود فایل ویدئویی و دریافت پیش‌بینی کامل BADAS-Open.
 */
export const postBadasFull = async (file: File): Promise<BadasFullPrediction> => {
    const formData = new FormData();
    formData.append('video', file);

    try {
        const response = await apiClient.post<BadasFullPrediction>('/predict/badas', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    } catch (error) {
        console.error("Error uploading video for full prediction:", error);
        throw new Error("Failed to get prediction from the backend.");
    }
};
