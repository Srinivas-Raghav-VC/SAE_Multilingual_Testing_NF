"""Model and SAE loading using sae_lens.

Provides GemmaWithSAE class that handles:
- Model loading (Gemma 2 2B)
- SAE loading (Gemma Scope)
- Hidden state extraction
- SAE activation computation
- Steered generation
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

from config import MODEL_ID, HIDDEN_DIM, ATTN_IMPLEMENTATION, HF_TOKEN


class GemmaWithSAE:
    """Gemma 2 model with Gemma Scope SAE access."""
    
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = dtype if self.device == "cuda" else torch.float32
        self.model = None
        self.tokenizer = None
        self.saes = {}  # layer -> SAE
    
    def load_model(self):
        """Load the base Gemma model."""
        print(f"Loading {MODEL_ID}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN if HF_TOKEN else None
        )
        
        # Configure based on available hardware
        model_kwargs = {
            "torch_dtype": self.dtype,
            "device_map": "auto",
        }
        
        # Try flash attention if available
        try:
            if self.device == "cuda":
                model_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION
        except Exception:
            print("Flash attention not available, using default")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN if HF_TOKEN else None,
            **model_kwargs
        )
        
        self.model.eval()
        print(f"Model loaded on {self.device}.")
        return self
    
    def load_sae(self, layer: int):
        """Load Gemma Scope SAE for a specific layer.
        
        Args:
            layer: Layer number (0-indexed)
            
        Returns:
            SAE object for that layer
        """
        if layer in self.saes:
            return self.saes[layer]
        
        print(f"Loading SAE for layer {layer}...")
        
        # Gemma Scope SAE naming convention
        sae, _, _ = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res-canonical",
            sae_id=f"layer_{layer}/width_16k/canonical",
            device=self.device,
        )
        
        self.saes[layer] = sae
        print(f"SAE loaded: {sae.cfg.d_sae} features")
        return sae
    
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
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Ensure steering vector is on correct device
        steering_vector = steering_vector.to(self.device).to(self.dtype)
        
        def hook(module, input, output):
            """Add steering to hidden states during forward pass."""
            hidden = output[0] if isinstance(output, tuple) else output
            # Add steering: shape (batch, seq, hidden) + (hidden,) broadcast
            steered = hidden + strength * steering_vector.unsqueeze(0).unsqueeze(0)
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
