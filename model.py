"""Model and SAE loading using sae_lens.

VERIFIED: SAE.from_pretrained syntax is correct per Gemma Scope README.
Release: gemma-scope-2b-pt-res-canonical
SAE ID format: layer_{N}/width_16k/canonical
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

from config import MODEL_ID, HIDDEN_DIM, SAE_RELEASE, ATTN_IMPLEMENTATION


class GemmaWithSAE:
    """Gemma 2 model with Gemma Scope SAE access."""
    
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        self.saes = {}  # layer -> SAE
    
    def load_model(self, attn_implementation=None):
        """Load Gemma 2 model with configurable attention implementation.
        
        Args:
            attn_implementation: "sdpa" (default), "flash_attention_2", or "eager"
                - sdpa: Scaled Dot-Product Attention (works on A100, recommended)
                - flash_attention_2: Requires flash-attn package
                - eager: Standard PyTorch attention (slower)
        """
        attn_impl = attn_implementation or ATTN_IMPLEMENTATION
        
        print(f"Loading {MODEL_ID}...")
        print(f"  Attention: {attn_impl}")
        print(f"  Dtype: {self.dtype}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        # Note: Gemma 2 uses HybridCache for sliding window attention
        # flash_attention_2 requires: pip install flash-attn --no-build-isolation
        model_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": "auto",
        }
        
        # Only add attn_implementation if not using default
        if attn_impl in ["flash_attention_2", "eager"]:
            model_kwargs["attn_implementation"] = attn_impl
        
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
        self.model.eval()
        print(f"Model loaded. Device: {next(self.model.parameters()).device}")
        return self
    
    def load_sae(self, layer):
        """Load Gemma Scope SAE for a specific layer.
        
        Uses sae-lens library with canonical SAEs (L0 ≈ 100).
        Verified syntax from google/gemma-scope-2b-pt-res README.
        """
        if layer in self.saes:
            return self.saes[layer]
        
        print(f"Loading SAE for layer {layer}...")
        
        # SAE_RELEASE should be "gemma-scope-2b-pt-res-canonical"
        # sae_id format: "layer_{N}/width_16k/canonical"
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=f"layer_{layer}/width_16k/canonical",
            device=self.device,
        )
        
        self.saes[layer] = sae
        print(f"  SAE loaded: {sae.cfg.d_sae} features, L0 ≈ {sparsity.get('l0', 'unknown') if sparsity else 'unknown'}")
        return sae
    
    def get_hidden_states(self, text, layer):
        """Get hidden states at a specific layer."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[layer].squeeze(0)  # (seq_len, hidden_dim)
    
    def get_sae_activations(self, text, layer):
        """Get SAE feature activations for text at layer."""
        hidden = self.get_hidden_states(text, layer)
        sae = self.load_sae(layer)
        # SAE expects float32 for stability
        return sae.encode(hidden.to(torch.float32))  # (seq_len, n_features)
    
    def generate_with_steering(self, prompt, layer, steering_vector, strength, max_new_tokens=50):
        """Generate with steering vector added at layer.
        
        Steering is applied via forward hook on the residual stream.
        The steering vector is added after the layer's computations.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Ensure steering vector is on correct device and dtype
        steer_vec = steering_vector.to(self.device, self.dtype)
        
        def hook(module, input, output):
            # Gemma 2 layer output is (hidden_states, ...) tuple
            hidden = output[0] if isinstance(output, tuple) else output
            # Add steering: broadcast over batch and sequence dimensions
            steered = hidden + strength * steer_vec.unsqueeze(0).unsqueeze(0)
            return (steered,) + output[1:] if isinstance(output, tuple) else steered
        
        # Register hook on the target layer
        handle = self.model.model.layers[layer].register_forward_hook(hook)
        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Greedy for reproducibility
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            return self.tokenizer.decode(out[0], skip_special_tokens=True)
        finally:
            handle.remove()


if __name__ == "__main__":
    print("Testing model and SAE loading...")
    print("=" * 50)
    
    model = GemmaWithSAE()
    model.load_model()
    
    # Test hidden states
    test_text = "Hello world"
    h = model.get_hidden_states(test_text, layer=10)
    print(f"\nHidden shape for '{test_text}': {h.shape}")
    
    # Test SAE
    acts = model.get_sae_activations(test_text, layer=10)
    print(f"SAE activations shape: {acts.shape}")
    print(f"Active features (>0): {(acts > 0).sum().item()}")
    print(f"L0 per token: {(acts > 0).float().sum(dim=1).mean().item():.1f}")
