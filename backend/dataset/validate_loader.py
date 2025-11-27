import torch
import logging
import os
from .nexar_loader import create_loader

# EN: Initialize logging
# FA: راه‌اندازی لاگینگ پایه
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def validate():
    """
    Validates the Nexar data loader by fetching one batch and printing its properties.
    """
    logging.info("Attempting to create and validate the data loader...")
    try:
        # Define the absolute path to the dataset
        # This assumes the script is run from backend/dataset
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        dataset_path = os.path.join(project_root, 'nexar_collision_prediction')

        logging.info(f"Using dataset path: {dataset_path}")

        # Create the training data loader
        train_loader = create_loader(
            split="train",
            batch_size=4,
            data_path=dataset_path,
            num_workers=0 # Set to 0 for debugging in main thread
        )

        # Get one batch of data
        logging.info("Fetching one batch of data from the loader...")
        frames_tensor, labels = next(iter(train_loader))
        logging.info("Successfully fetched one batch.")

        # Print the shape of the frames tensor
        logging.info(f"Shape of frames tensor (B, T, C, H, W): {frames_tensor.shape}")

        # Calculate and print the number of positive samples
        num_positive = torch.sum(labels["cls"]).item()
        logging.info(f"Number of positive samples in the batch: {num_positive}")

        # Print some metadata for debugging
        logging.info("Sample labels from the batch:")
        for i in range(frames_tensor.shape[0]):
             logging.info(f"  Sample {i}: cls={labels['cls'][i].item()}, tta={labels['tta'][i].item():.2f}")

    except Exception as e:
        logging.error(f"An error occurred during validation: {e}", exc_info=True)

if __name__ == "__main__":
    validate()
