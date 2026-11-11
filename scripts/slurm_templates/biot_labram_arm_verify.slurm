#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=biot_labram_arm_verify
#SBATCH --output=/users/hamehdi/projects/sleepfm-mesa/logs/biot_labram_arm_verify_%j.log

cd /users/hamehdi/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate

echo "=== Installing missing deps ==="
pip install --no-input einops
PIPEXIT=$?
if [ $PIPEXIT -ne 0 ]; then
    echo "FATAL: pip install einops failed with code $PIPEXIT"
    exit $PIPEXIT
fi

echo "=== Dependency versions ==="
python3 -c "
import torch, timm, einops
import linear_attention_transformer
print('torch', torch.__version__)
print('timm', timm.__version__)
print('einops', einops.__version__)
print('linear_attention_transformer OK')
print('cuda available:', torch.cuda.is_available())
"

echo "=== BIOT: import + forward pass with dummy data ==="
python3 -c "
import sys, torch
sys.path.insert(0, '/scratch/project_2019517/BIOT')
from model.biot import BIOTClassifier

CKPT = '/scratch/project_2019517/BIOT/pretrained-models/EEG-SHHS+PREST-18-channels.ckpt'
model = BIOTClassifier(emb_size=256, heads=8, depth=4, n_classes=5, n_fft=200, hop_length=100, n_channels=18)
state_dict = torch.load(CKPT, map_location='cpu')
model.biot.load_state_dict(state_dict)
model.eval()

device = torch.device('cuda')
model.to(device)
# dummy input: (batch, n_channels, time) at 200Hz, 10s window like BIOT's TUAB usage
x = torch.randn(2, 18, 2000, device=device)
with torch.no_grad():
    out = model(x)
print('BIOT forward OK, output shape:', out.shape)
assert out.shape == (2, 5), f'unexpected output shape {out.shape}'
print('BIOT ARM VERIFY: PASS')
"
BIOT_EXIT=$?

echo "=== LaBraM: import + forward pass with dummy data ==="
python3 -c "
import sys, torch
from collections import OrderedDict
sys.path.insert(0, '/scratch/project_2019517/LaBraM')
from modeling_finetune import labram_base_patch200_200

CKPT = '/scratch/project_2019517/LaBraM/checkpoints/labram-base.pth'
model = labram_base_patch200_200(num_classes=5, in_chans=1, qkv_bias=False, init_values=0.1)

checkpoint = torch.load(CKPT, map_location='cpu')
checkpoint_model = checkpoint['model']
new_dict = OrderedDict()
for k, v in checkpoint_model.items():
    if k.startswith('student.'):
        new_dict[k[len('student.'):]] = v
state_dict = model.state_dict()
for k in ['head.weight', 'head.bias']:
    if k in new_dict and new_dict[k].shape != state_dict[k].shape:
        del new_dict[k]
model.load_state_dict(new_dict, strict=False)
model.eval()

device = torch.device('cuda')
model.to(device)
# dummy input matching forward_features expectations: (batch, n_patches, patch_size) at 200Hz
# LaBraM patch = 1s @ 200Hz; use 16 patches (<=16 patch time_embed limit), 1 channel
x = torch.randn(2, 1, 16, 200, device=device)
input_chans = list(range(17))  # cls token slot + 16 channel-position slots (dummy, just for shape check)
with torch.no_grad():
    out = model(x, input_chans=input_chans[:1+1])
print('LaBraM forward OK, output shape:', out.shape)
print('LaBraM ARM VERIFY: PASS')
"
LABRAM_EXIT=$?

echo "=== SUMMARY ==="
echo "BIOT exit code: $BIOT_EXIT"
echo "LaBraM exit code: $LABRAM_EXIT"
if [ $BIOT_EXIT -ne 0 ] || [ $LABRAM_EXIT -ne 0 ]; then
    echo "FATAL: ARM verification failed for one or both models"
    exit 1
fi
echo "ARM_VERIFY_ALL_PASS"
