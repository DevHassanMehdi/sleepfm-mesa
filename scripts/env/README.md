# Environments

## `sleepfm_venv_arm` (`/projappl/project_2019517/sleepfm_venv_arm`, Python 3.12, aarch64/GH200)

`sleepfm_venv_arm_frozen.txt` — package list derived from
`site-packages/*.dist-info` directory names (not a true `pip freeze`:
the venv's own Python binary is wrapped in an aarch64-only Apptainer
container and cannot execute on this x86_64 login node — attempting
`pip freeze` directly fails with `FATAL: ... arm64 could not run on
the host's amd64`. Versions are accurate; this just lists them a
different way). Notably, `torch` itself is **not** in this list — it's
provided by the `python-pytorch/2.10` module, not pip-installed into
the venv.

Rebuild:
```bash
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
python -m venv /projappl/project_2019517/sleepfm_venv_arm
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
pip install -r requirements_roihu_arm.txt   # repo root
```

## `kaggle_venv_py312` (`/scratch/project_2019517/kaggle_venv_py312`, Python 3.12)

`kaggle_venv_py312_frozen.txt` — real `pip freeze` output, 96 packages.
This venv lives on `/scratch` (180-day purge clock) with no
requirements file previously tracked anywhere — this frozen list is
now the only record of exactly what's installed.

Rebuild:
```bash
module use /appl/modulefiles/manual/aida/x86_64   # or aarch64, matching target node
module load python-pytorch/2.10
python -m venv /scratch/project_2019517/kaggle_venv_py312
source /scratch/project_2019517/kaggle_venv_py312/bin/activate
pip install -r scripts/env/kaggle_venv_py312_frozen.txt
```

## Module loads in active use

```bash
# GPU work (GH200/aarch64 nodes)
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10

# CPU-only work (x86_64 login/small-partition nodes)
module use /appl/modulefiles/manual/aida/x86_64
module load python-pytorch/2.10

# Older scripts (download_nsrr.sh, submit_all_folds.sh, submit_experiment.sh)
module load python-data/3.12-31.03   # seen in some older templates
```

## `privatemodules`

`~/privatemodules/x86_64/` exists but is currently **empty** — no
custom modulefiles to capture.
