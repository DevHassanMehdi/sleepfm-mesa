#!/bin/bash
#SBATCH --account=project_2019517
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=labram_arm_verify_retry
#SBATCH --output=/users/hamehd/projects/sleepfm-mesa/logs/labram_arm_verify_retry_%j.log

cd /users/hamehd/projects/sleepfm-mesa
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate

echo "=== LaBraM: import + forward pass with dummy data (retry with weights_only=False) ==="
python3 -c "
import sys, torch
from collections import OrderedDict
sys.path.insert(0, '/scratch/project_2019517/LaBraM')
from modeling_finetune import labram_base_patch200_200

CKPT = '/scratch/project_2019517/LaBraM/checkpoints/labram-base.pth'
model = labram_base_patch200_200(num_classes=5, in_chans=1, qkv_bias=False, init_values=0.1)

checkpoint = torch.load(CKPT, map_location='cpu', weights_only=False)
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
x = torch.randn(2, 1, 16, 200, device=device)
input_chans = list(range(2))
with torch.no_grad():
    out = model(x, input_chans=input_chans)
print('LaBraM forward OK, output shape:', out.shape)
assert out.shape == (2, 5), f'unexpected output shape {out.shape}'
print('LaBraM ARM VERIFY: PASS')
"
echo "EXIT: $?"
