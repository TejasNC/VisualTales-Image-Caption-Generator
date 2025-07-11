import torch
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
from dataset import FlickrDataset
from utils import CapCollat, get_latest_checkpoint, calculate_bleu, Vocabulary
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
from models import Encoder, DecoderWithAttention
import warnings


def load_model(checkpoint_path, vocab_size, device):
    """
    Load trained encoder and decoder models from checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        vocab_size (int): Size of the vocabulary.
        device (torch.device): Device to load the models onto.

    Returns:
        tuple: (encoder, decoder) - The loaded models.
    """
    # Model Parameters
    embed_dim = 256
    attention_dim = 512
    decoder_dim = 512
    encoder_dim = 2048
    dropout = 0.5

    # Initialize models
    encoder = Encoder(encoded_image_size=14).to(device)
    decoder = DecoderWithAttention(
        attention_dim=attention_dim,
        embed_dim=embed_dim,
        decoder_dim=decoder_dim,
        vocab_size=vocab_size,
        encoder_dim=encoder_dim,
        dropout=dropout,
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    # Set to evaluation mode
    encoder.eval()
    decoder.eval()

    return encoder, decoder


def visualize_caption(image_path, caption, output_dir=None):
    """
    Visualize an image with its generated caption.

    Args:
        image_path (str): Path to the image.
        caption (str): Caption to display.
        output_dir (str, optional): Directory to save the visualization.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.title(caption, fontsize=12)
        plt.axis("off")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            image_name = os.path.basename(image_path)
            output_path = os.path.join(
                output_dir, f"{os.path.splitext(image_name)[0]}_caption.png"
            )
            plt.savefig(output_path)
            print(f"Visualization saved to {output_path}")
            plt.close()
        else:
            plt.show()
    except Exception as e:
        print(f"Error visualizing caption: {e}")


def evaluate_model(
    encoder,
    decoder,
    test_loader,
    vocab,
    device,
    num_samples=5,
    output_dir=None,
    image_dir=None,
):
    """
    Evaluate the model on the test set and calculate BLEU scores.

    Args:
        encoder (Encoder): Encoder model.
        decoder (DecoderWithAttention): Decoder model.
        test_loader (DataLoader): DataLoader for the test set.
        vocab (Vocabulary): Vocabulary object.
        device (torch.device): Device to run the evaluation on.
        num_samples (int): Number of samples to visualize.
        output_dir (str): Directory to save visualizations.
        image_dir (str): Directory containing the images.

    Returns:
        dict: Dictionary containing BLEU scores.
    """
    encoder.eval()
    decoder.eval()

    all_references = []
    all_hypotheses = []

    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Samples to visualize
    sample_count = 0

    with torch.no_grad():
        for i, (images, captions, caption_lengths) in enumerate(
            tqdm(test_loader, desc="Evaluating")
        ):
            images = images.to(device)

            # Get reference captions
            references = []
            for j, caption_length in enumerate(caption_lengths):
                caption = captions[j][:caption_length].cpu().tolist()
                ref = [
                    vocab.itos[idx]
                    for idx in caption
                    if idx
                    not in [
                        vocab.stoi["<PAD>"],
                        vocab.stoi["<SOS>"],
                        vocab.stoi["<EOS>"],
                        vocab.stoi["<UNK>"],
                    ]
                ]
                references.append(ref)

            # Generate captions
            hypotheses = []
            for j in range(images.size(0)):
                img_encoding = encoder(images[j].unsqueeze(0))
                generated_words = decoder.generate(img_encoding, vocab.stoi)
                hypotheses.append(generated_words)

                # Visualize samples
                if sample_count < num_samples and output_dir:
                    # Reconstruct image path from dataset
                    image_path = os.path.join(
                        image_dir,
                        test_loader.dataset.dataset.img_paths.iloc[
                            i * test_loader.batch_size + j
                        ],
                    )
                    caption = " ".join(generated_words)
                    ref_caption = " ".join(references[j])

                    plt.figure(figsize=(10, 8))
                    # Try to load and display the image
                    try:
                        img = Image.open(image_path).convert("RGB")
                        plt.imshow(img)
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")
                        plt.text(
                            0.5, 0.5, "Image not available", ha="center", va="center"
                        )

                    plt.title(
                        f"Generated: {caption}\nReference: {ref_caption}", fontsize=10
                    )
                    plt.axis("off")

                    output_path = os.path.join(output_dir, f"sample_{sample_count}.png")
                    plt.savefig(output_path)
                    plt.close()

                    sample_count += 1

            # Store for BLEU calculation
            all_references.extend(references)
            all_hypotheses.extend(hypotheses)

    # Calculate BLEU scores
    bleu_scores, _ = calculate_bleu(all_references, all_hypotheses)

    return bleu_scores


def main(args):
    """
    Main function to test the image captioning model.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Using device: {device}")

    # Try to load vocabulary from saved file
    vocab = None
    vocab_path = os.path.join(args.checkpoint_dir, "vocab", "vocab.pkl")

    if os.path.exists(vocab_path):
        print(f"Loading vocabulary from {vocab_path}")
        vocab = Vocabulary.load_vocab(vocab_path)

        # Data transforms
        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        if args.full_test:
            # Import dataset here if needed for testing
            from dataset import FlickrDataset

            # Initialize dataset with the loaded vocabulary
            dataset = FlickrDataset(
                targ_dir=args.image_dir,
                caps_file=args.captions_file,
                transforms=transform,
                freq_threshold=10,
                preloaded_vocab=vocab,
            )
    else:
        warnings.warn(
            f"No vocabulary file found at {vocab_path}. Loading dataset to build vocabulary."
        )
        # Fall back to loading dataset for vocabulary
        from dataset import FlickrDataset

        # Initialize dataset
        print("Loading dataset...")
        dataset = FlickrDataset(
            targ_dir=args.image_dir,
            caps_file=args.captions_file,
            transforms=transform,
            freq_threshold=10,
        )
        vocab = dataset.vocab

    # Create DataLoader for testing
    if args.full_test:
        # Split into train/test if testing on full dataset
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        _, test_dataset = random_split(dataset, [train_size, test_size])

        collate_fn = CapCollat(pad_seq=dataset.vocab.stoi["<PAD>"], batch_first=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
        )

    # Load model
    checkpoint_path = args.checkpoint or get_latest_checkpoint(args.checkpoint_dir)
    if not checkpoint_path:
        print("No checkpoint found. Please provide a valid checkpoint.")
        return

    encoder, decoder = load_model(checkpoint_path, len(dataset.vocab), device)

    # Test the model
    if args.full_test:
        print("Evaluating model on test set...")
        bleu_scores = evaluate_model(
            encoder,
            decoder,
            test_loader,
            dataset.vocab,
            device,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            image_dir=args.image_dir,
        )

        print("\nBLEU Scores:")
        for metric, score in bleu_scores.items():
            print(f"  {metric}: {score:.4f}")

        # Save results to file
        if args.output_dir:
            with open(os.path.join(args.output_dir, "bleu_scores.txt"), "w") as f:
                for metric, score in bleu_scores.items():
                    f.write(f"{metric}: {score:.4f}\n")

    # Test on a single image if provided
    if args.image_path:
        print(f"Generating caption for image: {args.image_path}")
        # Load and preprocess image
        try:
            image = Image.open(args.image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Generate caption
            with torch.no_grad():
                encoder_out = encoder(image_tensor)
                if args.visualize_attention:
                    words, attention_weights = decoder.generate(
                        encoder_out,
                        dataset.vocab.stoi,
                        beam_size=args.beam_size,
                        max_caption_len=args.max_len,
                        return_attention=True,
                    )
                else:
                    words = decoder.generate(
                        encoder_out,
                        dataset.vocab.stoi,
                        beam_size=args.beam_size,
                        max_caption_len=args.max_len,
                        return_attention=False,
                    )

            # Print and visualize caption
            caption = " ".join(words)
            print(f"Generated caption: {caption}")

            visualize_caption(args.image_path, caption, args.output_dir)

        except Exception as e:
            print(f"Error processing image {args.image_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test an image captioning model")
    parser.add_argument(
        "--image_path", type=str, help="Path to a single image to caption"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images/",
        help="Directory containing images",
    )
    parser.add_argument(
        "--captions_file",
        type=str,
        default="/kaggle/input/flickr-image-dataset/flickr30k_images/results.csv",
        help="Path to captions file",
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to a specific checkpoint file"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoint files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for testing"
    )
    parser.add_argument(
        "--beam_size", type=int, default=3, help="Beam size for caption generation"
    )
    parser.add_argument(
        "--max_len", type=int, default=50, help="Maximum caption length"
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of samples to visualize"
    )
    parser.add_argument(
        "--full_test", action="store_true", help="Evaluate on the full test set"
    )
    parser.add_argument(
        "--visualize_attention", action="store_true", help="Visualize attention weights"
    )
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=None,
        help="Path to the saved vocabulary file (default: checkpoints/vocab/vocab.pkl)",
    )

    args = parser.parse_args()

    # If a specific vocab_path is provided, use it
