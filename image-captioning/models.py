import torch
from torch import nn
import torchvision
from torchvision.models import resnet101, ResNet101_Weights


class Encoder(nn.Module):
    """
    Using a pre-trained ResNet-101 model as the encoder for image feature extraction.

    Args:
        encoded_image_size (int): The size of the encoded image (spatial dimensions).
    """

    def __init__(self, encoded_image_size=14, fine_tune=False):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(
            (encoded_image_size, encoded_image_size)
        )

        self.fine_tune(fine_tune)

    def forward(self, images):
        """
        Forward propagation.

        Args:
            images (torch.Tensor): Input images of shape (batch_size, 3, image_size, image_size).

        Returns:
            torch.Tensor: Encoded images of shape (batch_size, encoded_image_size, encoded_image_size, 2048).
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(
            out
        )  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(
            0, 2, 3, 1
        )  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=False):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        Args:
            fine_tune (bool): If True, allows fine-tuning of ResNet layers 2 through 4.
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network for generating attention weights.

    Args:
        encoder_dim (int): Feature size of encoded images.
        decoder_dim (int): Size of the decoder's RNN.
        attention_dim (int): Size of the attention network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(
            encoder_dim, attention_dim
        )  # Transform encoded image
        self.decoder_att = nn.Linear(
            decoder_dim, attention_dim
        )  # Transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # Calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        Args:
            encoder_out (torch.Tensor): Encoded images of shape (batch_size, num_pixels, encoder_dim).
            decoder_hidden (torch.Tensor): Previous decoder output of shape (batch_size, decoder_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Attention-weighted encoding and attention weights.
        """
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att + 1e-8)  # added epsilon for numerical stability
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder with Attention for generating captions.

    Args:
        attention_dim (int): Size of the attention network.
        embed_dim (int): Embedding size.
        decoder_dim (int): Size of the decoder's RNN.
        vocab_size (int): Size of the vocabulary.
        encoder_dim (int): Feature size of encoded images.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        attention_dim,
        embed_dim,
        decoder_dim,
        vocab_size,
        encoder_dim=2048,
        dropout=0.5,
    ):
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout_p = dropout  # dropout probability

        self.attention = Attention(
            encoder_dim, decoder_dim, attention_dim
        )  # Attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # Embedding layer
        self.dropout = nn.Dropout(p=self.dropout_p)
        self.decode_step = nn.LSTMCell(
            embed_dim + encoder_dim, decoder_dim, bias=True
        )  # Decoding LSTMCell
        self.init_h = nn.Sequential(
            nn.Linear(encoder_dim, decoder_dim), nn.Tanh()
        )  # Linear layer to initialize hidden state; added non linearity for stability
        self.init_c = nn.Sequential(
            nn.Linear(encoder_dim, decoder_dim), nn.Tanh()
        )  # Linear layer to initialize cell state; added non linearity for stability
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # Sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # Scores over vocabulary
        self.init_weights()  # Initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        Args:
            encoder_out (torch.Tensor): Encoded images of shape (batch_size, num_pixels, encoder_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Initial hidden state and cell state.
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        Args:
            encoder_out (torch.Tensor): Encoded images of shape (batch_size, enc_image_size, enc_image_size, encoder_dim).
            encoded_captions (torch.Tensor): Encoded captions of shape (batch_size, max_caption_length).
            caption_lengths (torch.Tensor): Caption lengths of shape (batch_size,) or (batch_size, 1).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[int], torch.Tensor, torch.Tensor]:
                - Predictions for vocabulary.
                - Sorted encoded captions.
                - Decode lengths.
                - Attention weights.
                - Sort indices.
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(
            batch_size, -1, encoder_dim
        )  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths
        # Handle both 1D and 2D caption_lengths tensors
        if caption_lengths.dim() > 1:
            caption_lengths = caption_lengths.squeeze(1)

        # Ensure caption lengths are at least 1 to avoid negative decode lengths
        caption_lengths = torch.clamp(caption_lengths, min=1)

        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(
            encoded_captions
        )  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # Decode lengths are actual lengths - 1, ensure they're not negative
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word prediction scores and alphas
        predictions = torch.zeros(
            batch_size, max(decode_lengths), vocab_size, device=encoder_out.device
        )

        alphas = torch.zeros(
            batch_size, max(decode_lengths), num_pixels, device=encoder_out.device
        )

        # Decode captions
        for t in range(max(decode_lengths)):
            batch_size_t = sum(
                [l > t for l in decode_lengths]
            )  # only process sequences where the current time step is less than the length of the sequence

            # Skip if no sequences to process at this time step
            if batch_size_t == 0:
                break

            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t]
            )
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # Gating scalar
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat(
                    [embeddings[:batch_size_t, t, :], attention_weighted_encoding],
                    dim=1,
                ),
                (h[:batch_size_t], c[:batch_size_t]),
            )  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    @torch.no_grad()
    def generate(
        self,
        encoder_out,
        word_map,
        beam_size=3,
        max_caption_len=50,
        return_attention=False,
    ):
        """
        Generate caption for an image using beam search.

        Args:
            encoder_out (torch.Tensor): Encoded image of shape (1, enc_image_size, enc_image_size, encoder_dim).
            word_map (dict): Mapping from word to index (must include < SOS > and <EOS> tokens).
            beam_size (int): Beam size for search.
            max_caption_len (int): Maximum length of generated caption.
            return_attention (bool): Whether to return attention weights for visualization.

        Returns:
            Union[List[str], Tuple[List[str], torch.Tensor]]:
                - Generated caption as a list of words.
                - Attention weights if return_attention is True.
        """
        device = encoder_out.device
        vocab_size = self.vocab_size
        start_token = word_map["< SOS >"]
        end_token = word_map["<EOS>"]

        # Flatten image
        encoder_out = encoder_out.view(1, -1, self.encoder_dim)
        num_pixels = encoder_out.size(1)

        # Expand the image tensor to beam size
        encoder_out = encoder_out.expand(beam_size, num_pixels, self.encoder_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)

        # Store sequences, scores, and attention weights if needed
        sequences = torch.full(
            (beam_size, 1), start_token, dtype=torch.long, device=device
        )
        top_scores = torch.zeros(beam_size, 1, device=device)

        # Store attention weights if needed
        all_alphas = (
            torch.zeros(beam_size, max_caption_len, num_pixels, device=device)
            if return_attention
            else None
        )

        complete_seqs = []
        complete_seqs_scores = []
        complete_seqs_alphas = [] if return_attention else None

        # Beam search
        for step in range(max_caption_len):
            embeddings = self.embedding(sequences[:, -1])

            # Apply attention mechanism
            awe, alpha = self.attention(encoder_out, h)

            # Store attention weights if needed
            if return_attention:
                all_alphas[:, step, :] = alpha

            gate = self.sigmoid(self.f_beta(h))
            awe = gate * awe

            # LSTM step
            h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
            scores = self.fc(h)
            scores = torch.log_softmax(scores, dim=1)
            scores = top_scores.expand_as(scores) + scores

            # Handle first step specially
            if step == 0:
                top_k_scores, top_k_words = scores[0].topk(beam_size, dim=0)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(beam_size, dim=0)

            # Convert to beam indices and token indices
            prev_seq_inds = top_k_words // vocab_size
            next_word_inds = top_k_words % vocab_size

            # Build next sequences
            sequences = torch.cat(
                [sequences[prev_seq_inds], next_word_inds.unsqueeze(1)], dim=1
            )

            # Update hidden states and scores
            h = h[prev_seq_inds]
            c = c[prev_seq_inds]
            encoder_out = encoder_out[prev_seq_inds]
            top_scores = top_k_scores.unsqueeze(1)

            # Update attention weights if tracking
            if return_attention:
                all_alphas = all_alphas[prev_seq_inds]

            # Handle completed sequences
            incomplete_inds = []
            for i, word_idx in enumerate(next_word_inds):
                if word_idx.item() == end_token:
                    complete_seqs.append(sequences[i].tolist())
                    complete_seqs_scores.append(top_k_scores[i].item())
                    if return_attention:
                        complete_seqs_alphas.append(
                            all_alphas[i, : step + 1].unsqueeze(0)
                        )
                else:
                    incomplete_inds.append(i)

            # Break if all sequences are complete
            if len(incomplete_inds) == 0:
                break

            # Keep only incomplete sequences
            sequences = sequences[incomplete_inds]
            h = h[incomplete_inds]
            c = c[incomplete_inds]
            encoder_out = encoder_out[incomplete_inds]
            top_scores = top_scores[incomplete_inds]
            if return_attention:
                all_alphas = all_alphas[incomplete_inds]

        # Handle case where no sequences were completed
        if len(complete_seqs) == 0:
            complete_seqs = sequences.tolist()
            complete_seqs_scores = top_scores.squeeze(1).tolist()
            if return_attention:
                complete_seqs_alphas = [
                    all_alphas[i].unsqueeze(0) for i in range(len(sequences))
                ]

        # Select best sequence
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        best_seq = complete_seqs[i]

        # Convert indices to words
        inv_word_map = {v: k for k, v in word_map.items()}
        words = []
        for idx in best_seq:
            word = inv_word_map[idx]
            if word == "< SOS >":
                continue
            if word == "<EOS>":
                break
            words.append(word)

        # Return with attention weights if requested
        if return_attention and complete_seqs_alphas:
            best_attention = complete_seqs_alphas[i]
            return words, best_attention

        return words
    
    @torch.no_grad()
    def greedy_generate(
        self,
        encoder_out,
        word_map,
        max_caption_len=50,
        return_attention=False,
    ):
        """
        Generate caption for an image using greedy decoding (no beam search).

        Args:
            encoder_out (torch.Tensor): Encoded image of shape (1, enc_image_size, enc_image_size, encoder_dim).
            word_map (dict): Mapping from word to index (must include < SOS > and <EOS> tokens).
            max_caption_len (int): Maximum length of generated caption.
            return_attention (bool): Whether to return attention weights for visualization.

        Returns:
            Union[List[int], Tuple[List[int], torch.Tensor]]:
                - Generated caption as a list of token indices.
                - Attention weights if return_attention is True.
        """
        device = encoder_out.device
        start_token = word_map["< SOS >"]
        end_token = word_map["<EOS>"]

        encoder_out = encoder_out.view(1, -1, self.encoder_dim)
        num_pixels = encoder_out.size(1)

        h, c = self.init_hidden_state(encoder_out)

        seq = [start_token]
        alphas = [] if return_attention else None

        for step in range(max_caption_len):
            prev_word = torch.tensor([seq[-1]], dtype=torch.long, device=device)
            embeddings = self.embedding(prev_word)

            awe, alpha = self.attention(encoder_out, h)
            if return_attention:
                alphas.append(alpha)

            gate = self.sigmoid(self.f_beta(h))
            awe = gate * awe

            h, c = self.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
            scores = self.fc(h)
            _, next_word = scores.max(dim=1)
            next_word = next_word.item()
            seq.append(next_word)

            if next_word == end_token:
                break

        if return_attention:
            alphas = torch.stack(alphas, dim=1)  # (1, seq_len, num_pixels)
            return seq, alphas
        return seq
