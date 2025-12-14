"""Model and SAE loading using sae_lens.

Provides GemmaWithSAE class that handles:
- Model loading (Gemma 2 2B or 9B)
- SAE loading (Gemma Scope) with configurable width and retry logic
- Hidden state extraction
- SAE activation computation
- Steered generation with comprehensive validation

Research Rigor Features:
- Retry logic for SAE loading (handles network failures gracefully)
- Configurable SAE width for different analysis granularities
- Validation of steering vector dimensions and layer indices
- Warnings for near-zero steering vectors
- Explicit dtype handling to avoid precision bugs
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

import os
import time
from collections import OrderedDict
from config import (
    MODEL_ID,
    MODEL_ID_9B,
    ATTN_IMPLEMENTATION,
    HF_TOKEN,
    SAE_RELEASE,
    SAE_RELEASE_2B,
    SAE_RELEASE_9B,
    SAE_WIDTH,
    SAE_WIDTH_2B,
    SAE_WIDTH_9B,
)


class GemmaWithSAE:
    """Gemma 2 model with Gemma Scope SAE access."""
    
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_id: str = MODEL_ID,
        sae_release: str = SAE_RELEASE,
        sae_width: str = SAE_WIDTH,
    ):
        # Optional global scaling mode: set USE_9B=1 to run any experiment
        # with Gemma 2 9B + corresponding Gemma Scope SAEs without modifying
        # per-experiment code.
        use_9b = str(os.environ.get("USE_9B", "0")).lower() in ("1", "true", "yes")
        if use_9b and model_id == MODEL_ID:
            model_id = MODEL_ID_9B
        if use_9b and sae_release in (SAE_RELEASE, SAE_RELEASE_2B):
            sae_release = SAE_RELEASE_9B
        if use_9b and sae_width in (SAE_WIDTH, SAE_WIDTH_2B):
            sae_width = SAE_WIDTH_9B

        self.device = device if torch.cuda.is_available() else "cpu"
        # Handle device strings like "cuda:0" or "cuda:1" correctly
        is_cuda = str(self.device).startswith("cuda")
        self.dtype = dtype if is_cuda else torch.float32
        self.model = None
        self.tokenizer = None
        # Cache of loaded SAEs. Some experiments probe many layers and can OOM
        # on shared GPUs if we keep every SAE on-device. Use SAE_CACHE_SIZE to
        # cap this (LRU eviction).
        self.saes: "OrderedDict[int, SAE]" = OrderedDict()
        self.model_id = model_id
        self.sae_release = sae_release
        self.sae_width = sae_width

    def _sae_cache_limit(self) -> int | None:
        """Return max number of SAEs to keep in memory (LRU).

        Default to a small cap (2) to prevent VRAM blow‑up when sweeping many
        layers. Users can override via SAE_CACHE_SIZE; 0 disables caching.
        """
        raw = os.environ.get("SAE_CACHE_SIZE", "").strip()
        if not raw:
            return 2  # sane default to avoid unbounded GPU cache
        try:
            return max(0, int(raw))
        except Exception:
            return 2
    
    def load_model(self):
        """Load the base Gemma model on a single device.

        We intentionally avoid ``device_map='auto'`` here because the
        experiments register custom hooks and expect all parameters and
        activations to live on the same device. On the user's A100 40GB,
        Gemma 2 2B and 9B comfortably fit in bf16 or fp32.
        """
        print(f"Loading {self.model_id}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=HF_TOKEN if HF_TOKEN else None,
        )

        # Configure attention implementation and dtype.
        model_kwargs = {
            "torch_dtype": self.dtype,
        }
        try:
            if str(self.device).startswith("cuda"):
                model_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION
        except Exception:
            print("Attention implementation config failed, using default.")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=HF_TOKEN if HF_TOKEN else None,
            **model_kwargs,
        )

        # Move the entire model to the chosen device to avoid CPU/CUDA
        # mismatches in later experiments.
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}.")
        return self
    
    def load_sae(self, layer: int, max_retries: int = 3, retry_delay: float = 2.0):
        """Load Gemma Scope SAE for a specific layer with retry logic.
        
        Args:
            layer: Layer number (0-indexed)
            max_retries: Maximum number of retry attempts for network failures
            retry_delay: Delay in seconds between retries
            
        Returns:
            SAE object for that layer
            
        Raises:
            RuntimeError: If SAE cannot be loaded after all retries
        """
        if layer in self.saes:
            # Mark as most recently used.
            self.saes.move_to_end(layer)
            return self.saes[layer]
        
        sae_id = f"layer_{layer}/width_{self.sae_width}/canonical"
        print(f"Loading SAE for layer {layer} from {self.sae_release} (width={self.sae_width})...")
        
        last_error = None
        for attempt in range(max_retries):
            try:
                # Gemma Scope SAE naming convention
                sae, _, _ = SAE.from_pretrained(
                    release=self.sae_release,
                    sae_id=sae_id,
                    device=self.device,
                )
                
                cache_limit = self._sae_cache_limit()
                if cache_limit is None or cache_limit > 0:
                    self.saes[layer] = sae
                    self.saes.move_to_end(layer)

                    # LRU eviction to reduce peak VRAM (especially for 9B runs).
                    if cache_limit is not None:
                        while len(self.saes) > cache_limit:
                            evicted_layer, evicted_sae = self.saes.popitem(last=False)
                            if str(os.environ.get("SAE_CACHE_VERBOSE", "0")).lower() in ("1", "true", "yes"):
                                print(f"[model] Evicted SAE layer {evicted_layer} from cache (SAE_CACHE_SIZE={cache_limit}).")
                            try:
                                del evicted_sae
                            except Exception:
                                pass
                        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                            torch.cuda.empty_cache()
                print(f"SAE loaded: {sae.cfg.d_sae} features")
                return sae
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"  SAE load attempt {attempt + 1} failed: {e}")
                    print(f"  Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
        
        raise RuntimeError(
            f"Failed to load SAE for layer {layer} after {max_retries} attempts. "
            f"SAE ID: {sae_id}, Release: {self.sae_release}. "
            f"Last error: {last_error}"
        )
    
    def get_hidden_states(self, text: str, layer: int):
        """Get hidden states at a specific layer.
        
        Args:
            text: Input text
            layer: Layer number
            
        Returns:
            Tensor of shape (seq_len, hidden_dim)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model has not been loaded. Call load_model() first.")

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hs = outputs.hidden_states
        if hs is None:
            raise RuntimeError("Model did not return hidden states despite output_hidden_states=True.")

        # HF models return hidden_states with length n_layers+1, where index 0
        # is the embedding output and index i+1 corresponds to transformer block i.
        try:
            n_blocks = len(self.model.model.layers)
        except Exception:
            n_blocks = None

        if n_blocks is not None and len(hs) == n_blocks + 1:
            idx = layer + 1
        else:
            idx = layer

        if not (0 <= idx < len(hs)):
            raise ValueError(
                f"Requested layer {layer} (mapped to hidden_states[{idx}]) is out of range "
                f"for hidden_states length {len(hs)}."
            )

        h = hs[idx].squeeze(0)  # (seq_len, hidden_dim)
        # Handle single-token case: ensure 2D output
        if h.dim() == 1:
            h = h.unsqueeze(0)  # (1, hidden_dim)
        return h
    
    def get_sae_activations(self, text: str, layer: int):
        """Get SAE feature activations for text at layer.
        
        Args:
            text: Input text
            layer: Layer number
            
        Returns:
            Tensor of shape (seq_len, n_features)
        """
        hidden = self.get_hidden_states(text, layer)
        sae = self.load_sae(layer)
        return sae.encode(hidden)  # (seq_len, n_features)
    
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate text without steering.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated completion text (excluding prompt)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = int(inputs["input_ids"].shape[1])
        
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_ids = out[0][prompt_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    def generate_with_steering(
        self, 
        prompt: str, 
        layer: int, 
        steering_vector: torch.Tensor, 
        strength: float,
        max_new_tokens: int = 50,
        schedule: str | None = None,
        decay: float = 0.9,
    ) -> str:
        """Generate with steering vector added at layer.
        
        Args:
            prompt: Input prompt
            layer: Layer to add steering
            steering_vector: Direction to steer (shape: hidden_dim)
            strength: Multiplier for steering
            max_new_tokens: Maximum tokens to generate
            schedule: Steering schedule for when to apply the vector.
                - "constant": apply on all forward passes (default)
                - "prompt_only": apply only when processing the prompt
                - "generation_only": apply only during token-by-token generation (CAA-style)
                - "exp_decay": apply during generation with exponential decay per token
            decay: Exponential decay factor for "exp_decay" schedule.
            
        Returns:
            Generated completion text (excluding prompt)
        """
        if self.model is None:
            raise ValueError("Base model has not been loaded. Call load_model() first.")

        # Sanity‑check the target layer index against the underlying
        # transformer stack. This prevents silent failures where a hook
        # is registered on a non‑existent layer.
        try:
            num_layers = len(self.model.model.layers)
        except AttributeError:
            raise RuntimeError(
                "Expected self.model.model.layers to be the transformer "
                "block stack, but this attribute was not found."
            )
        if not (0 <= layer < num_layers):
            raise ValueError(
                f"Requested layer {layer} is out of range for this model "
                f"(valid: 0–{num_layers-1})."
            )

        # Steering vector must be 1D for correct broadcasting in the hook.
        # Flatten if needed (e.g., (1, hidden) -> (hidden,)), but error if multi-row.
        if steering_vector.dim() > 1:
            if steering_vector.shape[0] == 1:
                steering_vector = steering_vector.squeeze(0)
            else:
                raise ValueError(
                    f"steering_vector must be 1D or (1, hidden_dim), got shape {steering_vector.shape}. "
                    "Cannot broadcast multi-row steering vectors."
                )

        # Sanity check: ensure the steering vector matches the model's hidden size
        expected_dim = getattr(self.model.config, "hidden_size", None)
        if expected_dim is not None and steering_vector.shape[-1] != expected_dim:
            raise ValueError(
                f"steering_vector has trailing dimension {steering_vector.shape[-1]}, "
                f"but model.hidden_size={expected_dim}."
            )

        # Check for zero or near-zero steering vector (would cause silent failure)
        steering_norm = torch.norm(steering_vector.float()).item()
        if steering_norm < 1e-6:
            raise ValueError(
                f"Steering vector has near-zero norm ({steering_norm:.2e}). "
                "Cannot proceed with steering - check feature selection or normalization."
            )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = int(inputs["input_ids"].shape[1])
        
        # Ensure steering vector is on correct device and detached from any
        # computation graph to avoid gradient leakage; dtype is aligned with
        # the hidden states inside the hook.
        steering_vector = steering_vector.detach().to(self.device)

        # Allow overriding steering schedule via env var without changing
        # experiment code.
        if schedule is None:
            schedule = os.environ.get("STEERING_SCHEDULE", "constant")
        schedule = str(schedule).strip().lower()
        if "STEERING_DECAY" in os.environ:
            try:
                decay = float(os.environ["STEERING_DECAY"])
            except ValueError:
                pass

        gen_step = 0
        
        def hook(module, input, output):
            """Add steering to hidden states during forward pass."""
            nonlocal gen_step
            hidden = output[0] if isinstance(output, tuple) else output
            # Work in the same dtype and device as the hidden states to avoid
            # float/bfloat16 mismatches and CUDA device errors.
            steer_vec = steering_vector.to(dtype=hidden.dtype, device=hidden.device)

            seq_len = int(hidden.shape[1])
            apply = True
            local_strength = strength

            if schedule == "prompt_only":
                apply = seq_len > 1
            elif schedule in ("generation_only", "gen_only"):
                apply = seq_len == 1
            elif schedule in ("exp_decay", "decay"):
                apply = seq_len == 1
                if apply:
                    local_strength = strength * (decay ** gen_step)
                    gen_step += 1
            elif schedule == "constant":
                apply = True
            else:
                # Unknown schedule: default to constant to avoid silent no-op.
                apply = True

            if not apply:
                return output

            steered = hidden + local_strength * steer_vec.unsqueeze(0).unsqueeze(0)
            return (steered,) + output[1:] if isinstance(output, tuple) else steered
        
        # Register hook on target layer
        handle = self.model.model.layers[layer].register_forward_hook(hook)
        
        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            gen_ids = out[0][prompt_len:]
            return self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        finally:
            # Always remove hook
            handle.remove()


if __name__ == "__main__":
    print("Testing GemmaWithSAE...")
    
    model = GemmaWithSAE()
    model.load_model()
    
    # Test hidden states
    h = model.get_hidden_states("Hello world", layer=10)
    print(f"Hidden shape: {h.shape}")  # Expected: (3, 2304) for "Hello world"
    
    # Test SAE
    acts = model.get_sae_activations("Hello world", layer=10)
    print(f"SAE acts shape: {acts.shape}")  # Expected: (3, 16384)
    print(f"Active features: {(acts > 0).sum().item()}")
    
    # Test generation
    output = model.generate("The capital of India is", max_new_tokens=20)
    print(f"Generation: {output}")
