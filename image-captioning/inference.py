import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import os
from utils import get_latest_checkpoint, Vocabulary
from models import Encoder, DecoderWithAttention
import warnings


def load_image(image_path, transform=None):
    """
    Load an image from a file path and apply transformations.

    Args:
        image_path (str): Path to the image file.
        transform (callable, optional): Transformations to apply to the image.

    Returns:
        torch.Tensor: Transformed image tensor.
    """
    try:
        image = Image.open(image_path).convert("RGB")

        if transform is None:
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )

        image = transform(image)
        return image.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def visualize_attention(image_path, caption, attention_map, save_path=None):
    """
    Visualize the attention weights for an image and caption.

    Args:
        image_path (str): Path to the image file.
        caption (list): List of words in the caption.
        attention_map (torch.Tensor): Attention weights tensor of shape [1, caption_length, num_pixels].
        save_path (str, optional): Path to save the visualization.
    """
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Resize attention map dimensions to match image
    attention_map = (
        attention_map.squeeze(0).cpu().numpy()
    )  # [caption_length, num_pixels]

    # Determine grid size for the plot (rows = 1 for image, 1 for full caption, and 1 for each word's attention)
    num_words = len(caption)
    plt.figure(figsize=(14, num_words * 2.5))

    # Plot original image
    plt.subplot(num_words + 2, 1, 1)
    plt.imshow(image)
    plt.title("Original Image", fontsize=12)
    plt.axis("off")

    # Plot caption
    plt.subplot(num_words + 2, 1, 2)
    plt.text(0.5, 0.5, " ".join(caption), horizontalalignment="center", fontsize=14)
    plt.axis("off")

    # Display attention maps for each word
    for i, word in enumerate(caption):
        plt.subplot(num_words + 2, 1, i + 3)

        # Reshape attention to match spatial dimensions
        att_map = attention_map[i].reshape(
            int(np.sqrt(attention_map.shape[1])), int(np.sqrt(attention_map.shape[1]))
        )

        # Resize attention map to match image size for visualization
        plt.imshow(image)
        plt.imshow(att_map, alpha=0.5, cmap="jet")
        plt.title(f"Attention for '{word}'", fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Try to load vocabulary from saved file
    vocab = None
    vocab_path = os.path.join(args.checkpoint_dir, "vocab", "vocab.pkl")

    if os.path.exists(vocab_path):
        print(f"Loading vocabulary from {vocab_path}")
        vocab = Vocabulary.load_vocab(vocab_path)
    else:
        warnings.warn(
            f"No vocabulary file found at {vocab_path}. Loading dataset to build vocabulary."
        )
        # Fall back to loading dataset for vocabulary
        from dataset import FlickrDataset

        dataset = FlickrDataset(
            targ_dir=args.image_dir, caps_file=args.captions_file, freq_threshold=5
        )
        vocab = dataset.vocab

    # Initialize models
    encoder = Encoder(encoded_image_size=14).to(device)
    decoder = DecoderWithAttention(
        attention_dim=512,
        embed_dim=256,
        decoder_dim=512,
        vocab_size=len(vocab),
        encoder_dim=2048,
        dropout=0.5,
    ).to(device)

    # Load checkpoint
    checkpoint_path = args.checkpoint or get_latest_checkpoint(args.checkpoint_dir)
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
    else:
        print("No checkpoint found. Using untrained model.")

    # Set models to evaluation mode
    encoder.eval()
    decoder.eval()

    # Load image
    image_tensor = load_image(args.image_path)
    if image_tensor is None:
        return

    image_tensor = image_tensor.to(device)

    # Generate caption
    with torch.no_grad():
        encoder_out = encoder(image_tensor)
        if args.visualize_attention:
            words, attention_weights = decoder.generate(
                encoder_out,
                vocab.stoi,
                beam_size=args.beam_size,
                max_caption_len=args.max_len,
                return_attention=True,
            )
        else:
            words = decoder.generate(
                encoder_out,
                vocab.stoi,
                beam_size=args.beam_size,
                max_caption_len=args.max_len,
                return_attention=False,
            )

    # Display results
    caption = " ".join(words)
    print(f"Generated caption: {caption}")

    # Visualize attention if requested
    if args.visualize_attention:
        os.makedirs(args.output_dir, exist_ok=True)
        image_name = os.path.basename(args.image_path)
        output_path = os.path.join(
            args.output_dir, f"{os.path.splitext(image_name)[0]}_attention.png"
        )
        visualize_attention(
            args.image_path, words, attention_weights, save_path=output_path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate image captions using a trained model"
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint file",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoint files",
    )
    parser.add_argument(
        "--beam_size", type=int, default=3, help="Beam size for caption generation"
    )
    parser.add_argument(
        "--max_len", type=int, default=50, help="Maximum caption length"
    )
    parser.add_argument(
        "--visualize_attention", action="store_true", help="Visualize attention weights"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images/",
        help="Path to the directory containing images",
    )
    parser.add_argument(
        "--captions_file",
        type=str,
        default="/kaggle/input/flickr-image-dataset/flickr30k_images/results.csv",
        help="Path to the file containing captions",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=None,
        help="Path to the saved vocabulary file (default: checkpoints/vocab/vocab.pkl)",
    )

    args = parser.parse_args()

    # If a specific vocab_path is provided, use it
    if args.vocab_path:
        args.checkpoint_dir = os.path.dirname(os.path.dirname(args.vocab_path))

    main(args)
