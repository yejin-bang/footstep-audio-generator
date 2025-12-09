| Tensor Type | What It Is | Shape Example | Purpose |
|-------------|-----------|---------------|---------|
| `conditioning` | Text prompt dict | `[{"prompt": "...", ...}]` | Input to conditioner |
| `conditioning_tensors` | Text embeddings | `[1, 77, 768]` | Guide diffusion (NOT audio!) |
| `init_audio` (input) | Raw audio waveform | `[1, 2, 264600]` | Starting point for variations |
| `init_audio` (after encode) | Audio latent | `[1, 64, 828]` | Encoded starting point |
| `noise` | Random latent | `[1, 64, 828]` | Starting noise for generation |
| `sampled` (before decode) | Denoised latent | `[1, 64, 828]` | Generated latent (VAE space) |
| `sampled` (after decode) | Final audio | `[1, 2, 264600]` | Output waveform |



# BEFORE LoRA creation:
model = get_pretrained_model("stabilityai/stable-audio-open-1.0")
# model structure:
StableAudioModel (1B parameters)
├── pretransform (VAE encoder/decoder)
├── conditioner (T5 text encoder)
└── model (DiT transformer - THIS IS WHAT GETS WRAPPED)
    ├── timestep_features
    ├── transformer
    │   ├── blocks[0]
    │   │   ├── norm1
    │   │   ├── attn
    │   │   │   ├── to_q: Linear(768, 768)  ← Will be wrapped with LoRA
    │   │   │   ├── to_k: Linear(768, 768)  ← Will be wrapped with LoRA
    │   │   │   ├── to_v: Linear(768, 768)  ← Will be wrapped with LoRA
    │   │   │   └── to_out: Linear(768, 768) ← Will be wrapped with LoRA
    │   │   ├── norm2
    │   │   └── ff
    │   │       ├── net[0]: Linear(768, 3072) ← Will be wrapped with LoRA
    │   │       └── net[2]: Linear(3072, 768) ← Will be wrapped with LoRA
    │   ├── blocks[1]
    │   │   └── ... (same structure)
    │   └── ... (24 blocks total in Stable Audio)
    └── final_layer

# AFTER LoRA creation:
lora = create_lora_from_config(config, model)
# lora structure:
LoRAWrapper
├── target_model: StableAudioModel (reference to base model above)
├── target_map: dict of all Linear layers found
│   └── "model/transformer/blocks/0/attn/to_q": {
│       "module": Linear(768, 768),  # Original layer
│       "parent": attn  # Parent module
│   }
└── net: LoRANetwork
    └── lora_modules: dict of LoRA wrappers
        └── "model/transformer/blocks/0/attn/to_q": LoRALinear
            ├── original_module: Linear(768, 768)  # Reference to base model layer
            ├── lora_down: Linear(768, 16)  # NEW! Your trainable weights
            └── lora_up: Linear(16, 768)  # NEW! Your trainable weights

# AFTER lora.activate():
# The base model's Linear layers are REPLACED with LoRA wrappers
# So when you call model.forward(), it uses LoRA!