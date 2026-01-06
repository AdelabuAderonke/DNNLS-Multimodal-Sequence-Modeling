# DNNLS-Multimodal-Sequence-Modeling Final Assessment 
**Author:** Aderonke Adelabu Kaozara

Neural architecture for visual story reasoning using multimodal sequence modeling. Integrates visual encoder, text encoder, sequence model, attention mechanism, and dual decoders to generate both image and text predictions for narrative continuation.

---

## Introduction and Problem Statement

This repository contains the final assessment for the **Deep Neural Networks and Learning Systems (DNNLS)** course. The objective of this work is to build on an existing storytelling model through the incorporation of Transformer-based architectures and cross-modal attention layers, then evaluating how these additions improve reasoning capabilities and narrative coherence.
The task focuses on **story reasoning**, where a model must understand causal and temporal relationships between events in short stories and generate consistent narrative outcomes.

- Dataset: https://huggingface.co/datasets/daniel3303/StoryReasoning
- Folder Structure Overview

  - Final_assessment(Base).ipynb: Contains the baseline architecture the original implementation before improvements.
  - experiment_assessment_updated.ipynb: Contains the improved architecture — with added enhancements like transformer-based encoding and cross-modal attention.
  - results/ (folder): Stores all experiment outputs including,generated images and text samples, architecture diagrams illustrating both baseline and improved models
### Baseline Architecture:
- Visual Encoder(CNN based image)
- TextEncoder(LSTM)
- Simple Attention
- Sequence model

### Problem Definition

Given a baseline sequence-to-sequence storytelling model, we aim to:
- **Transformer architecture:**  Replace RNN components with a multi-head attention mechanism to better capture long range story dependencies and event relationships.

- **Cross-modal attention:**  Implement a joint attention module that explicitly connects visual and textual features, allowing the model to reason about story context across both modalities.


Performance is evaluated using both quantitative metrics and qualitative analysis of generated stories.

---


### Model Architecture Overview
Starting from the baseline encoder-decoder architecture, a SharpCrossModal model with the following key components:
- Transformer Text Encoder : This replaces LSTM with self-attention mechanism
- Cross-Modal Attention : Fuses visual and textual features
- U-Net Visual Encoder/Decoder : Preserves spatial details with skip connections

A high-level diagram of the modified architecture can be found here:

```
results/architectural_diagram.png
```

### Transformer Text Encoder
The proposed text encoder uses multi-head self-attention to capture long-range dependencies in text descriptions:

```python
class TransformerTextEncoder(nn.Module):
   """
   Transformer-based text encoder - replaces LSTM encoder
   Uses self-attention to capture long-range dependencies better
   """
   def __init__(self, vocab_size, embedding_dim, hidden_dim, num_heads=4, num_layers=2, dropout=0.1):
       super().__init__()
       self.vocab_size = vocab_size
       self.embedding_dim = embedding_dim
       self.hidden_dim = hidden_dim


       # Token embedding
       self.embedding = nn.Embedding(vocab_size, embedding_dim)


       # Positional encoding (learnable)
       self.pos_embedding = nn.Parameter(torch.randn(1, 512, embedding_dim))


       # Transformer encoder layers
       encoder_layer = nn.TransformerEncoderLayer(
           d_model=embedding_dim,
           nhead=num_heads,
           dim_feedforward=hidden_dim * 2,
           dropout=dropout,
           batch_first=True
       )
       self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


       # Project to desired hidden dimension
       self.projection = nn.Linear(embedding_dim, hidden_dim)


   def forward(self, input_seq):
       """
       Args:
           input_seq: (batch_size, seq_len)
       Returns:
           encoded: (batch_size, seq_len, hidden_dim)
           pooled: (batch_size, hidden_dim) - for compatibility
       """
       batch_size, seq_len = input_seq.shape


       # Embed tokens
       embedded = self.embedding(input_seq)  # (B, L, E)


       # Add positional encoding
       embedded = embedded + self.pos_embedding[:, :seq_len, :]


       # Apply transformer
       encoded = self.transformer(embedded)  # (B, L, E)


       # Project to hidden_dim
       encoded = self.projection(encoded)  # (B, L, H)


       # Pool for compatibility (mean pooling)
       pooled = encoded.mean(dim=1)  # (B, H)


       return encoded, pooled

```

---

### Cross-Modal Attention Mechanism
The cross-modal attention allows image features to attend to text tokens, enabling visual predictions conditioned on textual context:
```python

class SharpCrossModal(nn.Module):
   def __init__(self, text_encoder, max_text_length=50):
       super().__init__()
       self.text_encoder = text_encoder
       self.max_text_length = max_text_length


       # ENCODER with separate layers for skip connections
       # Level 1: 60x125 -> 30x62
       self.enc1_1 = nn.Conv2d(3, 32, 3, padding=1)
       self.enc1_2 = nn.Conv2d(32, 32, 3, padding=1)
       self.pool1 = nn.MaxPool2d(2)


       # Level 2: 30x62 -> 15x31
       self.enc2_1 = nn.Conv2d(32, 64, 3, padding=1)
       self.enc2_2 = nn.Conv2d(64, 64, 3, padding=1)
       self.pool2 = nn.MaxPool2d(2)


       # Level 3: 15x31 (bottleneck)
       self.enc3_1 = nn.Conv2d(64, 128, 3, padding=1)
       self.enc3_2 = nn.Conv2d(128, 128, 3, padding=1)


       self.leaky = nn.LeakyReLU(0.2)


       # Text projection - now projects each token, not just pooled
       self.text_proj = nn.Linear(256, 128)


       # Cross-attention: image queries attend to text keys/values
       self.cross_attention = nn.MultiheadAttention(128, 8, batch_first=True)


       # Layer norm for stability
       self.norm1 = nn.LayerNorm(128)
       self.norm2 = nn.LayerNorm(128)


       # Feed-forward network after attention
       self.ffn = nn.Sequential(
           nn.Linear(128, 256),
           nn.GELU(),
           nn.Dropout(0.1),
           nn.Linear(256, 128)
       )


       # DECODER with skip connections
       self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
       self.dec1_1 = nn.Conv2d(128 + 64, 64, 3, padding=1)  # +64 from skip
       self.dec1_2 = nn.Conv2d(64, 64, 3, padding=1)


       self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
       self.dec2_1 = nn.Conv2d(64 + 32, 32, 3, padding=1)   # +32 from skip
       self.dec2_2 = nn.Conv2d(32, 32, 3, padding=1)


       # Final output
       self.final = nn.Sequential(
           nn.Conv2d(32, 16, 3, padding=1),
           nn.LeakyReLU(0.2),
           nn.Conv2d(16, 3, 3, padding=1),
           nn.Tanh()
       )


       # ===== TEXT PREDICTION COMPONENTS =====
       # Simpler approach: Direct mapping from visual features to text


       # Project fused features to text space
       self.visual_to_text = nn.Sequential(
           nn.Linear(128, 256),
           nn.ReLU(),
           nn.Dropout(0.1),
           nn.Linear(256, 256)
       )


       # Simple text decoder (no LSTM, just transformer-style)
       self.text_attention = nn.MultiheadAttention(256, 4, batch_first=True)
       self.text_norm = nn.LayerNorm(256)
       self.text_ffn = nn.Sequential(
           nn.Linear(256, 512),
           nn.ReLU(),
           nn.Dropout(0.1),
           nn.Linear(512, 256)
       )


       # Output projection to vocabulary
       self.text_output = nn.Linear(256, text_encoder.vocab_size)


   def forward(self, frames, text_inputs):
       batch_size = frames.size(0)


       # Process ALL 4 frames, not just frame 4
       all_frame_features = []
       for i in range(4):
           frame = frames[:, i, :, :, :]


           # Encode each frame
           x = self.leaky(self.enc1_1(frame))
           x = self.leaky(self.enc1_2(x))
           x = self.pool1(x)


           x = self.leaky(self.enc2_1(x))
           x = self.leaky(self.enc2_2(x))
           x = self.pool2(x)


           x = self.leaky(self.enc3_1(x))
           x = self.leaky(self.enc3_2(x))


           all_frame_features.append(x)


       # Concatenate or average all frames
       img_features = torch.stack(all_frame_features, dim=1).mean(dim=1)  # [batch, 128, 15, 31]


       # For skip connections, use the last frame
       frame4 = frames[:, -1, :, :, :]
       x = self.leaky(self.enc1_1(frame4))
       skip1 = self.leaky(self.enc1_2(x))
       x = self.pool1(skip1)


       x = self.leaky(self.enc2_1(x))
       skip2 = self.leaky(self.enc2_2(x))
       x = self.pool2(skip2)


       # Encode ALL text descriptions, not just text 4
       all_text_features = []
       for i in range(4):
           text = text_inputs[:, i, :]
           text_encoded, _ = self.text_encoder(text)
           text_proj = self.text_proj(text_encoded)
           all_text_features.append(text_proj)


       # Concatenate all text along sequence dimension
       text_tokens = torch.cat(all_text_features, dim=1)  # [batch, seq_len*4, 128]


       # Flatten spatial features
       spatial_tokens = img_features.flatten(2).permute(0, 2, 1)


       # Cross-attention
       attended, attn_weights = self.cross_attention(
           query=spatial_tokens,
           key=text_tokens,
           value=text_tokens
       )


       # Residual + norm + FFN
       spatial_tokens = self.norm1(spatial_tokens + attended)
       ffn_out = self.ffn(spatial_tokens)
       fused_features = self.norm2(spatial_tokens + ffn_out)  # [batch, 465, 128]


       # ===== IMAGE PREDICTION =====
       # Reshape back
       x = fused_features.permute(0, 2, 1).view(batch_size, 128, 15, 31)


       # DECODER with skip connections
       # Level 1: 15x31 -> 30x62
       x = self.up1(x)
       x = torch.cat([x, skip2], dim=1)
       x = self.leaky(self.dec1_1(x))
       x = self.leaky(self.dec1_2(x))


       # Level 2: 30x62 -> 60x124 (then pad/interpolate to match skip1)
       x = self.up2(x)
       if x.shape[-2:] != skip1.shape[-2:]:
           x = F.interpolate(x, size=skip1.shape[-2:], mode='bilinear', align_corners=False)
       x = torch.cat([x, skip1], dim=1)
       x = self.leaky(self.dec2_1(x))
       x = self.leaky(self.dec2_2(x))


       # Final output
       predicted_image = self.final(x)


       # Ensure exact size
       if predicted_image.shape[-2:] != (60, 125):
           predicted_image = F.interpolate(predicted_image, size=(60, 125), mode='bilinear', align_corners=False)


       # ===== TEXT PREDICTION =====
       # SIMPLEST APPROACH: Almost direct copy of frame 4 text
       # Just learn a small residual modification


       # Get frame 4's original text tokens (not features)
       frame4_text_tokens = text_inputs[:, -1, :]  # [batch, seq_len]


       # Encode it again
       frame4_encoded, _ = self.text_encoder(frame4_text_tokens)  # [batch, seq_len, 256]


       # Project to output dimension
       frame4_projected = self.visual_to_text(self.text_proj(frame4_encoded))  # [batch, seq_len, 256]


       # Add a TINY visual context as a residual
       visual_context = fused_features.mean(dim=1, keepdim=True)  # [batch, 1, 128]
       visual_residual = self.visual_to_text(visual_context)  # [batch, 1, 256]


       # Add small residual (scaled down)
       text_features = frame4_projected + 0.1 * visual_residual


       # Pad or truncate to max_text_length
       if text_features.size(1) < self.max_text_length:
           padding = torch.zeros(batch_size, self.max_text_length - text_features.size(1), 256,
                               device=text_features.device)
           text_features = torch.cat([text_features, padding], dim=1)
       else:
           text_features = text_features[:, :self.max_text_length, :]


       # Project to vocabulary - this should almost reconstruct frame 4's text
       text_logits = self.text_output(text_features)  # [batch, max_len, vocab_size]


       return predicted_image, text_logits

```


## Results

### Quantitative Evaluation

- Loss curves: `results/final_loss_curves.png`

### Qualitative Analysis

Example generated stories and comparisons are shown in:

- `results/final_sample.png`
- `results/final_sample_2.png`

---

## Conclusions


### FINAL STATISTICS
- Starting Train Loss: 17.4962
- Final Train Loss: 5.4489
- Improvement: 68.9%
- Final Image Loss: 0.2369
- Final Text Loss: 1.8005
- Final Validation Loss: 0.3551
- Overfitting Gap: 5.0939
---

The key improvement that was done on the baseline architecture are:
- Enhanced Text Processing: The traditional LSTMs was replaced with Transformer-based text encoding, improving semantic understanding     and capturing longer-range textual dependencies.
- Multi-modal Fusion: Introduces the cross-modal attention between visual and textual features, enabling context-aware interaction and - better integration of information from both modalities.
- Improved Visual Quality:Added a U-Net architecture with skip connections in the visual decoder, preserving spatial details and      reducing blurring in generated video frames.

## Challenges

- Blurry images : Some details of the predicted images are still unclear or fuzzy.
- Texts repetition : Also the predicted texts sometimes say the same word too many times.


---

## Future Work

- Implement GAN-based training

- Extend to variable-length video sequences
- Add perceptual loss (VGG-based)
- Implement proper autoregressive text generation with teacher forcing

## References
- Chen et al., "Cross-Modal Attention for Multi-Modal Learning" (2020)
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- StoryReasoning Dataset: https://huggingface.co/datasets/daniel3303/StoryReasoning
- Vaswani et al., "Attention Is All You Need" (2017) - Transformer architecture





