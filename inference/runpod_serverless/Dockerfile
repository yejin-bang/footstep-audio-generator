# ============================================================================
# RunPod Serverless - Footstep Audio Generator Dockerfile
# ============================================================================
# 
# This Dockerfile builds a production-ready serverless environment for
# generating footstep audio using LoRA fine-tuned Stable Audio Open.
# 
# Author: Yejin Bang
# ============================================================================

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# ============================================================================
# LAYER 1: System dependencies and pip upgrade
# ============================================================================

RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip

# ============================================================================
# LAYER 2: Upgrade PyTorch to 2.5.1 (Required for Stable Audio)
# ============================================================================

RUN pip uninstall -y torch torchvision torchaudio && \
    pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ============================================================================
# LAYER 3: Clone required repositories
# ============================================================================

RUN git clone https://github.com/NeuralNotW0rk/LoRAW.git

# ============================================================================
# LAYER 4: Install Python packages
# ============================================================================

RUN pip install --no-cache-dir stable-audio-tools

RUN cd /app/LoRAW && \
    pip install --no-cache-dir -e .

RUN pip install --no-cache-dir \
    wandb \
    huggingface-hub \
    runpod

# ============================================================================
# LAYER 5: Install Flash Attention 2 (Critical for performance)
# ============================================================================

RUN pip install --no-cache-dir ninja packaging && \
    MAX_JOBS=4 pip install --no-cache-dir flash-attn --no-build-isolation

# ============================================================================
# LAYER 6: Apply Critical Bug Fixes
# ============================================================================

# Bug Fix 1: Fix import in LoRAW gradio.py
RUN sed -i 's/from stable_audio_tools\.training\.utils import copy_state_dict/from stable_audio_tools.models.utils import copy_state_dict/g' \
    /app/LoRAW/interface/gradio.py

# Bug Fix 2: Fix checkpoint loading in stable-audio-tools
# Note: This may not be needed with pip-installed version, but keeping for safety
RUN python3 -c "import site; import re; \
pkg_path = site.getsitepackages()[0] + '/stable_audio_tools/models/utils.py'; \
content = open(pkg_path, 'r').read(); \
old_pattern = r'state_dict = torch\.load\(ckpt_path, map_location=\"cpu\", weights_only=True\)\[\"state_dict\"\]'; \
new_code = '''ckpt = torch.load(ckpt_path, map_location=\"cpu\", weights_only=True)\n        if \"state_dict\" in ckpt:\n            state_dict = ckpt[\"state_dict\"]\n        else:\n            state_dict = ckpt'''; \
content = re.sub(old_pattern, new_code, content); \
open(pkg_path, 'w').write(content)" || echo "Bug fix skipped - may not be needed"

# ============================================================================
# LAYER 7: Copy application files
# ============================================================================


COPY model_config_lora.json /app/model_config_lora.json


COPY best.ckpt /app/best.ckpt


COPY serverless_handler.py /app/serverless_handler.py

# ============================================================================
# LAYER 8: Verify installation
# ============================================================================

RUN python3 -c "import torch; \
import stable_audio_tools; \
from loraw.network import LoRAWrapper; \
print('✓ All imports successful'); \
print(f'✓ PyTorch: {torch.__version__}'); \
print(f'✓ CUDA available: {torch.cuda.is_available()}')"

# ============================================================================
# CONFIGURATION
# ============================================================================


CMD ["python", "-u", "/app/serverless_handler.py"]