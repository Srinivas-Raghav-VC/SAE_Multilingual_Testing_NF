"""Model and SAE loading using sae_lens."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

from config import MODEL_ID, HIDDEN_DIM


class GemmaWithSAE:
    """Gemma 2 model with Gemma Scope SAE access."""
    
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        self.saes = {}  # layer -> SAE
    
    def load_model(self):
        print(f"Loading {MODEL_ID}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=self.dtype,
            device_map="auto",
        )
        self.model.eval()
        print("Model loaded.")
        return self
    
    def load_sae(self, layer):
        """Load Gemma Scope SAE for a specific layer."""
        if layer in self.saes:
            return self.saes[layer]
        
        print(f"Loading SAE for layer {layer}...")
        sae, _, _ = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res-canonical",
            sae_id=f"layer_{layer}/width_16k/canonical",
            device=self.device,
        )
        self.saes[layer] = sae
        print(f"SAE loaded: {sae.cfg.d_sae} features")
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
        return sae.encode(hidden)  # (seq_len, n_features)
    
    def generate_with_steering(self, prompt, layer, steering_vector, strength, max_new_tokens=50):
        """Generate with steering vector added at layer."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        def hook(module, input, output):
            # Add steering to hidden states
            hidden = output[0] if isinstance(output, tuple) else output
            hidden = hidden + strength * steering_vector.unsqueeze(0).unsqueeze(0)
            return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
        
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
            handle.remove()


if __name__ == "__main__":
    model = GemmaWithSAE()
    model.load_model()
    
    # Test hidden states
    h = model.get_hidden_states("Hello world", layer=10)
    print(f"Hidden shape: {h.shape}")
    
    # Test SAE
    acts = model.get_sae_activations("Hello world", layer=10)
    print(f"SAE acts shape: {acts.shape}")
    print(f"Active features: {(acts > 0).sum().item()}")
