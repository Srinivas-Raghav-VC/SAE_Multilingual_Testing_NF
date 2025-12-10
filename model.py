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

import time
from config import MODEL_ID, HIDDEN_DIM, ATTN_IMPLEMENTATION, HF_TOKEN, SAE_RELEASE, SAE_WIDTH


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
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = dtype if self.device == "cuda" else torch.float32
        self.model = None
        self.tokenizer = None
        self.saes = {}  # layer -> SAE
        self.model_id = model_id
        self.sae_release = sae_release
        self.sae_width = sae_width
    
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
            if self.device == "cuda":
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
                
                self.saes[layer] = sae
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
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        return outputs.hidden_states[layer].squeeze(0)  # (seq_len, hidden_dim)
    
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
            Generated text (including prompt)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
    
    def generate_with_steering(
        self, 
        prompt: str, 
        layer: int, 
        steering_vector: torch.Tensor, 
        strength: float,
        max_new_tokens: int = 50
    ) -> str:
        """Generate with steering vector added at layer.
        
        Args:
            prompt: Input prompt
            layer: Layer to add steering
            steering_vector: Direction to steer (shape: hidden_dim)
            strength: Multiplier for steering
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text (including prompt)
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

        # Optional but cheap sanity check: ensure the steering vector matches
        # the model's hidden size so that broadcasting behaves as intended.
        expected_dim = getattr(self.model.config, "hidden_size", None)
        if expected_dim is not None and steering_vector.shape[-1] != expected_dim:
            raise ValueError(
                f"steering_vector has trailing dimension {steering_vector.shape[-1]}, "
                f"but model.hidden_size={expected_dim}."
            )

        # Check for zero or near-zero steering vector (silent failure risk)
        if torch.norm(steering_vector.float()) < 1e-6:
            import warnings
            warnings.warn(
                "Steering vector has near-zero norm. "
                "Generation will proceed without effective steering."
            )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Ensure steering vector is on correct device; dtype is aligned with
        # the hidden states inside the hook.
        steering_vector = steering_vector.to(self.device)
        
        def hook(module, input, output):
            """Add steering to hidden states during forward pass."""
            hidden = output[0] if isinstance(output, tuple) else output
            # Work in the same dtype as the hidden states to avoid
            # float/bfloat16 mismatches downstream.
            steer_vec = steering_vector.to(hidden.dtype)
            steered = hidden + strength * steer_vec.unsqueeze(0).unsqueeze(0)
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
            return self.tokenizer.decode(out[0], skip_special_tokens=True)
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
