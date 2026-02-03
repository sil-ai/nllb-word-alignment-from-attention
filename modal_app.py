"""
Modal app for GPU-based NLLB attention extraction.

Deploy with:
    modal deploy modal_app.py

This creates the 'nllb-alignment' Modal app with the AlignmentExtractor class.
"""

import modal

app = modal.App(
    "nllb-alignment",
    image=modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "sentencepiece>=0.1.99",
        "accelerate>=0.24.0",
        "numpy>=1.24.0",
    )
)

nllb_cache_vol = modal.Volume.from_name("nllb-model-cache", create_if_missing=True)
NLLB_CACHE_PATH = "/root/model_cache"


@app.cls(
    gpu="T4",
    image=app.image,
    timeout=1800,
    secrets=[
        modal.Secret.from_name("my-huggingface-secret"),
        modal.Secret.from_dict({"HF_HOME": NLLB_CACHE_PATH}),
    ],
    volumes={NLLB_CACHE_PATH: nllb_cache_vol},
)
class AlignmentExtractor:
    """GPU-based NLLB attention extractor for word alignment."""

    model_id: str = modal.parameter(default="facebook/nllb-200-distilled-600M")
    layer: int = modal.parameter(default=3)

    @modal.enter()
    def load_model(self):
        """Load model once when container starts."""
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.model_available = False
        print(f"Loading model {self.model_id}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id,
                output_attentions=True,
                torch_dtype=torch.float16,
            )
            self.model = self.model.cuda()
            self.model.eval()
            self.model_available = True
            print("Model loaded!")
        except OSError as e:
            print(f"Warning: Model {self.model_id} not found: {e}")
            print("Alignment extraction will be skipped.")
            self.tokenizer = None
            self.model = None

    @modal.method()
    def extract_batch(self, batch: dict) -> dict:
        """
        Process a batch of sentence pairs and return attention matrices.

        Args:
            batch: {
                "idx": batch_index,
                "src_texts": [...],  # Source/reference texts
                "tgt_texts": [...],  # Target/revision texts
                "layer": int,        # Attention layer to extract (default 3)
            }

        Returns:
            {"idx": batch_index, "results": [...], "model_available": bool}
            where each result has src_tokens, tgt_tokens, attention_matrix
        """
        import numpy as np
        import torch

        # If model failed to load, return empty results with flag
        if not self.model_available:
            return {
                "idx": batch["idx"],
                "results": [],
                "model_available": False,
            }

        src_texts = batch["src_texts"]
        tgt_texts = batch["tgt_texts"]
        layer = batch.get("layer", self.layer)

        # Use eng_Latn as default - the actual language doesn't matter for tokenization
        # since NLLB uses a shared multilingual vocabulary and we filter out language tokens
        default_lang = "eng_Latn"

        results = []

        for src_text, tgt_text in zip(src_texts, tgt_texts):
            if not src_text or not tgt_text:
                results.append({
                    "src_tokens": [],
                    "tgt_tokens": [],
                    "attention_matrix": [],
                })
                continue

            try:
                # Tokenize source (language setting only affects the prepended language token,
                # which we filter out anyway when mapping tokens to words)
                self.tokenizer.src_lang = default_lang
                src_encoded = self.tokenizer(
                    src_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=512,
                )
                input_ids = src_encoded["input_ids"].cuda()
                attention_mask = src_encoded["attention_mask"].cuda()

                # Tokenize target for teacher forcing
                # labels = the tokens we want to predict (what the model outputs)
                # decoder_input_ids = shifted right (what the model sees as input)
                self.tokenizer.src_lang = default_lang
                tgt_encoded = self.tokenizer(
                    tgt_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=512,
                )
                labels = tgt_encoded["input_ids"]
                # Shift labels right: prepend decoder_start_token_id, remove last token
                # This creates proper teacher forcing: decoder sees [START, tok1, tok2, ...]
                # while predicting [tok1, tok2, ..., EOS]
                decoder_start_token_id = self.model.config.decoder_start_token_id
                shifted = torch.cat([
                    torch.tensor([[decoder_start_token_id]]),
                    labels[:, :-1]
                ], dim=1)
                decoder_input_ids = shifted.cuda()

                # Forward pass
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        output_attentions=True,
                        return_dict=True,
                    )

                # Extract cross-attention from specified layer
                cross_attn = outputs.cross_attentions[layer]
                attn_matrix = cross_attn[0].mean(dim=0).cpu().numpy().astype(np.float32)

                # Get token strings
                # Return labels tokens (what's being predicted) not decoder_input_ids
                src_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
                tgt_tokens = self.tokenizer.convert_ids_to_tokens(labels[0].tolist())

                results.append({
                    "src_tokens": src_tokens,
                    "tgt_tokens": tgt_tokens,
                    "attention_matrix": attn_matrix.tolist(),
                })

            except Exception as e:
                print(f"Error processing pair: {e}")
                results.append({
                    "src_tokens": [],
                    "tgt_tokens": [],
                    "attention_matrix": [],
                    "error": str(e),
                })

        return {"idx": batch["idx"], "results": results, "model_available": True}

    @modal.method()
    def extract_batch_all_layers(self, batch: dict) -> dict:
        """
        Process a batch of sentence pairs and return attention matrices for ALL layers.

        This is more efficient for layer evaluation since we only run one forward pass
        per sentence pair instead of one per layer.

        Args:
            batch: {
                "idx": batch_index,
                "src_texts": [...],  # Source/reference texts
                "tgt_texts": [...],  # Target/revision texts
                "layers": [0, 1, 2, ...],  # Which layers to extract
            }

        Returns:
            {"idx": batch_index, "results": [...], "model_available": bool}
            where each result has src_tokens, tgt_tokens, attention_matrices (dict by layer)
        """
        import numpy as np
        import torch

        # If model failed to load, return empty results with flag
        if not self.model_available:
            return {
                "idx": batch["idx"],
                "results": [],
                "model_available": False,
            }

        src_texts = batch["src_texts"]
        tgt_texts = batch["tgt_texts"]
        layers = batch.get("layers", list(range(11)))  # Default: layers 0-10

        # Use eng_Latn as default - the actual language doesn't matter for tokenization
        default_lang = "eng_Latn"

        results = []

        for src_text, tgt_text in zip(src_texts, tgt_texts):
            if not src_text or not tgt_text:
                results.append({
                    "src_tokens": [],
                    "tgt_tokens": [],
                    "attention_matrices": {},
                })
                continue

            try:
                # Tokenize source
                self.tokenizer.src_lang = default_lang
                src_encoded = self.tokenizer(
                    src_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=512,
                )
                input_ids = src_encoded["input_ids"].cuda()
                attention_mask = src_encoded["attention_mask"].cuda()

                # Tokenize target for teacher forcing
                # labels = the tokens we want to predict (what the model outputs)
                # decoder_input_ids = shifted right (what the model sees as input)
                self.tokenizer.src_lang = default_lang
                tgt_encoded = self.tokenizer(
                    tgt_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=512,
                )
                labels = tgt_encoded["input_ids"]
                # Shift labels right: prepend decoder_start_token_id, remove last token
                # This creates proper teacher forcing: decoder sees [START, tok1, tok2, ...]
                # while predicting [tok1, tok2, ..., EOS]
                decoder_start_token_id = self.model.config.decoder_start_token_id
                shifted = torch.cat([
                    torch.tensor([[decoder_start_token_id]]),
                    labels[:, :-1]
                ], dim=1)
                decoder_input_ids = shifted.cuda()

                # Single forward pass - gets all layers at once
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        output_attentions=True,
                        return_dict=True,
                    )

                # Extract cross-attention from ALL requested layers
                attention_matrices = {}
                for layer in layers:
                    if layer < len(outputs.cross_attentions):
                        cross_attn = outputs.cross_attentions[layer]
                        attn_matrix = cross_attn[0].mean(dim=0).cpu().numpy().astype(np.float32)
                        attention_matrices[layer] = attn_matrix.tolist()

                # Get token strings
                # Return labels tokens (what's being predicted) not decoder_input_ids
                src_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
                tgt_tokens = self.tokenizer.convert_ids_to_tokens(labels[0].tolist())

                results.append({
                    "src_tokens": src_tokens,
                    "tgt_tokens": tgt_tokens,
                    "attention_matrices": attention_matrices,
                })

            except Exception as e:
                print(f"Error processing pair: {e}")
                results.append({
                    "src_tokens": [],
                    "tgt_tokens": [],
                    "attention_matrices": {},
                    "error": str(e),
                })

        return {"idx": batch["idx"], "results": results, "model_available": True}
