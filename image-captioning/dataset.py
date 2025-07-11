import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
import logging
from torchvision import transforms
from utils import Vocabulary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlickrDataset(Dataset):
    """
    Args:
        targ_dir (str): Directory containing the images.
        caps_file (str): Path to the CSV file with image paths and captions.
        transforms (callable, optional): Transformations to apply to the images.
        freq_threshold (int): Minimum frequency for a word to be included in the vocabulary.
        preloaded_vocab (Vocabulary, optional): A preloaded vocabulary to use instead of building one.
    """

        self.targ_dir = targ_dir
        self.df = pd.read_csv(caps_file, delimiter='|')
        self.transforms = transforms

        # Ensure column names are stripped of whitespace
        self.df.columns = self.df.columns.str.strip()
        self.img_paths = self.df['image_name']
        self.captions = self.df['comment'] 

        # Build vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Retrieve an image and its corresponding caption.
    
        Args:
            idx (int): Index of the sample to retrieve.
    
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The image and its numericalized caption.
        """
        caption = self.captions.iloc[idx]
        image_path = self.img_paths.iloc[idx]
    
        full_img_path = os.path.join(self.targ_dir, image_path)
    
        try:
            image = Image.open(full_img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image at {full_img_path}: {e}")
            # Return a placeholder image and caption
            return torch.zeros(3, 224, 224), torch.tensor([self.vocab.stoi["<UNK>"]])
    
        # Define default transformations if none are provided
        if self.transforms:
            image = self.transforms(image)
        else:
            # Resize to 224x224 and convert to tensor
            default_transforms = Compose([Resize((224, 224)), ToTensor()])
            image = default_transforms(image)
    
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption += [self.vocab.stoi["<EOS>"]]
    
        return image, torch.tensor(numericalized_caption)