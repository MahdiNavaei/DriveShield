import os
import logging
from typing import Dict, Any, Tuple, List

import decord
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F, transforms

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_nexar_local(path: str = "nexar_collision_prediction") -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads the Nexar dataset from a local directory using the correct file and column names.
    """
    logging.info(f"Attempting to load dataset from: {path}")
    data = {"train": [], "test_public": [], "test_private": []}
    
    base_path = os.path.abspath(path)
    if not os.path.isdir(base_path):
        logging.error(f"Dataset root directory not found: {base_path}")
        return data

    for split in ["train", "test-public", "test-private"]:
        split_path = os.path.join(base_path, split)
        if not os.path.isdir(split_path):
            continue

        for label_str in ["positive", "negative"]:
            label_dir = os.path.join(split_path, label_str)
            metadata_file = os.path.join(label_dir, "metadata.csv")

            if os.path.isfile(metadata_file):
                logging.info(f"Reading metadata from: {metadata_file}")
                df = pd.read_csv(metadata_file)
                
                for _, row in df.iterrows():
                    video_file = os.path.join(label_dir, row['file_name'])
                    if os.path.exists(video_file):
                        data[split.replace("-", "_")].append({
                            "video_path": video_file,
                            "label": 1 if label_str == "positive" else 0,
                            "time_of_event": row.get("time_of_event", -1),
                            "time_of_alert": row.get("time_of_alert", -1),
                            "metadata": row.to_dict()
                        })

    if all(not val for val in data.values()):
        logging.error("No data was loaded. Please check the dataset path and structure.")
    else:
        logging.info("Successfully loaded data splits.")

    return data

class NexarWindowDataset(Dataset):
    """
    EN: PyTorch Dataset for Nexar collision prediction using sliding windows.
    FA: دیتاست PyTorch برای پیش‌بینی تصادف Nexar با استفاده از پنجره‌های اسلایدینگ.
    """
    def __init__(self, samples: List[Dict[str, Any]], window_size: int = 60, stride: int = 15, fps: int = 30):
        """
        EN: Initializes the dataset.
        FA: دیتاست را مقداردهی اولیه می‌کند.
        """
        self.samples = samples
        self.window_size = window_size
        self.stride = stride
        self.fps = fps
        self.windows = self._create_windows()

    def _create_windows(self) -> List[Dict[str, Any]]:
        windows = []
        logging.info("Creating sliding windows for all samples...")
        for i, sample in enumerate(self.samples):
            try:
                video_reader = decord.VideoReader(sample["video_path"], num_threads=1)
                num_frames = len(video_reader)
                
                for start_frame in range(0, num_frames - self.window_size + 1, self.stride):
                    end_frame = start_frame + self.window_size
                    window_end_ts = end_frame / self.fps
                    
                    y_cls, y_tta = self._calculate_labels(sample, window_end_ts)

                    windows.append({
                        "sample_index": i,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "y_cls": y_cls,
                        "y_tta": y_tta,
                    })
            except decord.DECORDError as e:
                logging.error(f"Failed to read video: {sample['video_path']}. Error: {e}")
        logging.info(f"Created {len(windows)} windows from {len(self.samples)} samples.")
        return windows

    def _calculate_labels(self, sample: Dict[str, Any], window_end_ts: float) -> Tuple[int, float]:
        if sample["label"] == 1:  # Positive sample
            time_of_alert = sample.get("time_of_alert", 0)
            time_of_event = sample.get("time_of_event", 0)
            
            y_cls = 1 if time_of_alert <= window_end_ts <= time_of_event else 0
            y_tta = max(time_of_event - window_end_ts, 0.0)
        else:  # Negative sample
            y_cls = 0
            y_tta = 6.0
        return y_cls, y_tta

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # EN: Define the transform inside __getitem__ to ensure it's available in workers
        # FA: تعریف تبدیل در داخل __getitem__ برای اطمینان از در دسترس بودن آن در workerها
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True)
        ])

        window_info = self.windows[idx]
        sample = self.samples[window_info["sample_index"]]
        
        video_reader = decord.VideoReader(sample["video_path"], num_threads=1)
        
        frame_indices = list(range(window_info["start_frame"], window_info["end_frame"]))
        frames = video_reader.get_batch(frame_indices).asnumpy()
        
        # EN: Apply transform to each frame
        # FA: اعمال تبدیل روی هر فریم
        frames_tensor = torch.stack([transform(frame) for frame in frames])
        
        labels = {
            "cls": torch.tensor(window_info["y_cls"], dtype=torch.long),
            "tta": torch.tensor(window_info["y_tta"], dtype=torch.float32),
        }
        
        return frames_tensor, labels

def create_loader(split: str = "train", batch_size: int = 4, data_path: str = "./nexar_collision_prediction", num_workers: int = 4) -> DataLoader:
    """
    Creates a DataLoader for the Nexar dataset.

    Args:
        split (str): The dataset split to load ("train", "test_public", "test_private").
        batch_size (int): The batch size for the DataLoader.
        data_path (str): The root directory of the dataset.
        num_workers (int): The number of worker processes for data loading.

    Returns:
        A DataLoader instance for the specified dataset split.
    """
    logging.info(f"Loading data for split: {split}")
    all_data = load_nexar_local(data_path)
    split_data = all_data.get(split.replace("-", "_"))
    
    if not split_data:
        raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(all_data.keys())}")

    dataset = NexarWindowDataset(split_data)
    
    pin_memory = torch.cuda.is_available()
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=(split == "train")
    )
    
    logging.info(f"DataLoader created for split: {split} with {len(dataset)} windows.")
    return loader
