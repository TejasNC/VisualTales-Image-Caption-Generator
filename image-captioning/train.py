import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import os
from tqdm import tqdm
import numpy as np
from dataset import FlickrDataset
from utils import *
from models import *

"""
    Majority of the code is taken from : https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/tree/master
"""

# Paths to your dataset
image_dir = "/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images/"
captions_file = "/kaggle/input/flickr-image-dataset/flickr30k_images/results.csv"

# Model Parameters
batch_size = 32
embed_dim = 256  # Dimension of word embeddings
attention_dim = 512  # Dimension of linear layers in attention
decoder_dim = 512  # Dimension of decoder LSTM
encoder_dim = 2048  # = resnet.fc.in_features
dropout = 0.5

# Training Parameters
learning_rate = 1e-4
num_epochs = 50
grad_clip = 5.0
# Define the device globally and use it consistently
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Checkpoint Parameters
checkpoint_dir = "checkpoints"  # Directory to save model checkpoints

# Ensure checkpoint directory exists
os.makedirs(checkpoint_dir, exist_ok=True)


def train_one_epoch(
    encoder,
    decoder,
    train_loader,
    criterion,
    encoder_optimizer,
    decoder_optimizer,
    grad_clip,
    device,
):
    """
    Train the model for one epoch.

    Args:
        encoder: Encoder model.
        decoder: Decoder model.
        train_loader: DataLoader for the training set.
        criterion: Loss function.
        encoder_optimizer: Optimizer for the encoder.
        decoder_optimizer: Optimizer for the decoder.
        grad_clip: Gradient clipping value.
        device: Device to run the training on.

    Returns:
        float: Average training loss for the epoch.
    """
    encoder.train()
    decoder.train()
    epoch_loss = 0

    for batch_idx, (images, captions, caption_lengths) in enumerate(train_loader):
        # Move data to device
        images = images.to(device)
        captions = captions.to(device)
        caption_lengths = caption_lengths.to(device)

        # Forward pass
        encoder_out = encoder(images)
        predictions, encoded_captions, decode_lengths, alphas, sort_ind = decoder(
            encoder_out, captions, caption_lengths
        )

        # Reshape predictions and targets for loss calculation
        targets = encoded_captions[:, 1:]  # Exclude <SOS>
        predictions = predictions.view(-1, predictions.size(2))
        targets = targets.contiguous().view(-1)

        # Compute loss
        loss = criterion(predictions, targets)

        # Backward pass
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)

        # Update weights
        encoder_optimizer.step()
        decoder_optimizer.step()

        # Accumulate loss
        epoch_loss += loss.item()

        # Logging
        if (batch_idx + 1) % 50 == 0:
            print(
                f"Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
            )

    # Average loss for the epoch
    return epoch_loss / len(train_loader)


def validate(encoder, decoder, val_loader, criterion, device, word_map=None):
    """
    Validate the model on the validation set.

    Args:
        encoder: Encoder model.
        decoder: Decoder model.
        val_loader: DataLoader for the validation set.
        criterion: Loss function.
        device: Device to run the validation on.
        word_map: Dictionary mapping words to indices (optional, for BLEU calculation)

    Returns:
        float: Average validation loss.
        dict: BLEU scores (if word_map is provided)
    """
    encoder.eval()
    decoder.eval()
    val_loss = 0

    # For BLEU score calculation
    all_references = []
    all_hypotheses = []

    with torch.no_grad():
        for i, (images, captions, caption_lengths) in enumerate(val_loader):
            # Move data to device
            images = images.to(device)
            captions = captions.to(device)
            caption_lengths = caption_lengths.to(device)

            # Forward pass
            encoder_out = encoder(images)
            predictions, encoded_captions, decode_lengths, alphas, sort_ind = decoder(
                encoder_out, captions, caption_lengths
            )

            # Reshape predictions and targets for loss calculation
            targets = encoded_captions[:, 1:]  # Exclude < SOS >
            predictions = predictions.view(-1, predictions.size(2))
            targets = targets.contiguous().view(-1)

            # Compute loss
            loss = criterion(predictions, targets)
            val_loss += loss.item()

            # Calculate BLEU scores for a subset of validation data
            if word_map is not None and i % 5 == 0:  # Every 5th batch for efficiency
                # Get the original captions (reference)
                references = []
                for j, caption_length in enumerate(caption_lengths):
                    caption = captions[j][:caption_length].cpu().tolist()
                    # Convert indices to words, exclude special tokens
                    ref = [
                        word_map.itos[idx]
                        for idx in caption
                        if idx
                        not in [
                            word_map.stoi["<PAD>"],
                            word_map.stoi["< SOS >"],
                            word_map.stoi["<EOS>"],
                            word_map.stoi["<UNK>"],
                        ]
                    ]
                    references.append(ref)

                # Generate captions for the batch
                hypotheses = []
                for j in range(images.size(0)):
                    img_encoding = encoder(images[j].unsqueeze(0))
                    generated_words = decoder.generate(img_encoding, word_map.stoi)
                    hypotheses.append(generated_words)

                # Store for later BLEU calculation
                all_references.extend(references)
                all_hypotheses.extend(hypotheses)

    # Average validation loss
    val_loss = val_loss / len(val_loader)

    # Calculate BLEU scores if references and hypotheses are available
    bleu_scores = None
    if word_map is not None and all_references and all_hypotheses:
        bleu_scores, _ = calculate_bleu(all_references, all_hypotheses)

    return val_loss, bleu_scores


def main():
    """
    Main function to train the image captioning model.
    """
    # Dataset and DataLoader
    dataset = FlickrDataset(
        targ_dir=image_dir, caps_file=captions_file, freq_threshold=5
    )

    # Save vocabulary for later use during inference
    vocab_dir = os.path.join(checkpoint_dir, "vocab")
    os.makedirs(vocab_dir, exist_ok=True)
    vocab_path = os.path.join(vocab_dir, "vocab.pkl")
    dataset.vocab.save_vocab(vocab_path)
    print(f"Saved vocabulary to {vocab_path}")

    collate_fn = CapCollat(pad_seq=dataset.vocab.stoi["<PAD>"], batch_first=True)

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders for train and validation sets
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # Initialize models
    # Pass the device to the Encoder
    encoder = Encoder(encoded_image_size=14).to(device)
    decoder = DecoderWithAttention(
        attention_dim=attention_dim,
        embed_dim=embed_dim,
        decoder_dim=decoder_dim,
        vocab_size=len(dataset.vocab),
        encoder_dim=encoder_dim,
        dropout=dropout,
    ).to(
        device
    )  # Remove device parameter and move to device after initialization

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"]).to(device)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(
        decoder.parameters(), lr=learning_rate * 4
    )  # Higher learning rate for decoder

    checkpoint_path = get_latest_checkpoint(checkpoint_dir)
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        # Handle device placement for checkpoint loading
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
        else:
            # Load to CPU if CUDA is not available
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        start_epoch = checkpoint["epoch"]
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer_state_dict"])
        decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer_state_dict"])
    else:
        start_epoch = 0
        print("No checkpoint found, starting from scratch.")

    # Training loop
    for epoch in tqdm(range(start_epoch, num_epochs)):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")

        # Train for one epoch
        train_loss = train_one_epoch(
            encoder,
            decoder,
            train_loader,
            criterion,
            encoder_optimizer,
            decoder_optimizer,
            grad_clip,
            device,
        )
        print(f"Training Loss: {train_loss:.4f}")

        # Validate with BLEU score calculation
        val_loss, bleu_scores = validate(
            encoder,
            decoder,
            val_loader,
            criterion,
            device,
            word_map=dataset.vocab if epoch % 5 == 0 else None,
        )

        print(f"Validation Loss: {val_loss:.4f}")

        # Print BLEU scores if available
        if bleu_scores:
            print(f"BLEU Scores:")
            for metric, score in bleu_scores.items():
                print(f"  {metric}: {score:.4f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"model_epoch_{epoch + 1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "encoder_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "encoder_optimizer_state_dict": encoder_optimizer.state_dict(),
                    "decoder_optimizer_state_dict": decoder_optimizer.state_dict(),
                    "loss": train_loss,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved at {checkpoint_path}")


if __name__ == "__main__":
    main()
