# Migration Handoff

Written 2026-11-11. Nothing in this document was run as a live command beyond
what was needed to gather the facts below. No secrets, keys, or token
contents appear anywhere in this file, by path only where relevant.

## Current cluster state

**squeue** (user `hamehdi`): one entry, the interactive dashboard session
itself (`sys/dash`, job `3008984`, running on `rc5183`). No other compute
jobs queued or running — effectively empty.

**git**: current commit `d9c4cee` ("Flatten results/ into 8 top-level .md
files, verify every number, archive raw artifacts"), branch `main`, 2
commits ahead of `origin/main` (`d9c4cee`, `b993c21`), neither pushed.
`git config user.email` is still the placeholder `your-email@example.com`
— **needs a real value** before any future commit, since no real email
was supplied this session.

## Where everything lives

| What | Path | Notes |
|---|---|---|
| Repo clone | `/users/hamehdi/projects/sleepfm-mesa` | this repo |
| MESA/MrOS/SHHS data root | `/scratch/project_2019517/sleepfm-data/` | group `pepr_hamehdi`, **not** `project_2019517` |
| Moody/PhysioNet 2026 (I0006) area | `/scratch/project_2019517/moody2026_pilot/` | group `project_2019517` (correct), owner-only perms underneath |
| `sleepfm_venv_arm` | `/projappl/project_2019517/sleepfm_venv_arm` | group `project_2019517` |
| `kaggle_venv_py312` | `/scratch/project_2019517/kaggle_venv_py312` | group `project_2019517` |
| Safety-net archive (this task) | `/projappl/project_2019517/archive/results_raw_2026-11-11.tar.gz` | 1.7 MB, verified 1498/1498 files |
| Logs archive (prior task) | `/projappl/project_2019517/archive/logs_2026-11-11.tar.gz` | 98 MB, verified 714/714 files |
| NSRR downloader binary | `/users/hamehdi/.local/share/x86_64/gem/ruby/3.4.0/bin/nsrr` | installed under home, not under miniconda3 |
| NSRR gem environment | `~/.local/share/x86_64/gem/ruby/3.4.0/` | `gem environment gemdir` points here |
| NSRR download token | `.env` at repo root (`NSRR_TOKEN=...`) | path only, value not reproduced here |
| `miniconda3` (scratch) | `/scratch/project_2019517/miniconda3/` | has a cached but unused `nsrr-8.0.0.gem`; the real NSRR install is under home, not here |

**Sizes / free space** (2026-11-11): `/scratch/project_2019517` 1.9T used
of 2.0T, 147G free (93% used). `/projappl/project_2019517` 1.2G used of
15G, 14G free. Home (`/users/hamehdi`) 1.9G used of 15G, 14G free.

**File count by group, under `/scratch/project_2019517`**: 106,110 files
group `pepr_hamehdi` (gid 10036657), 106,771 files group `project_2019517`
(gid 2019517) — roughly even split. The split is driven mostly by
`sleepfm-data/` (group `pepr_hamehdi`) versus everything else (already
`project_2019517`).

**Recommended group change**: `sleepfm-data/` is currently group
`pepr_hamehdi` and should move to `project_2019517` once `hamehd` joins
the project, so the new account can read it. `moody2026_pilot/` is
**already** `project_2019517` — no change needed there (an earlier
assumption that both needed changing was wrong; checked directly this
session, not assumed).

## Environments and rebuild commands

**`sleepfm_venv_arm`** (`/projappl/project_2019517/sleepfm_venv_arm`,
Python 3.12, aarch64/GH200):
```bash
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10
python -m venv /projappl/project_2019517/sleepfm_venv_arm
source /projappl/project_2019517/sleepfm_venv_arm/bin/activate
pip install -r requirements_roihu_arm.txt   # repo root
```

**`kaggle_venv_py312`** (`/scratch/project_2019517/kaggle_venv_py312`,
Python 3.12):
```bash
module use /appl/modulefiles/manual/aida/x86_64   # or aarch64, matching target node
module load python-pytorch/2.10
python -m venv /scratch/project_2019517/kaggle_venv_py312
source /scratch/project_2019517/kaggle_venv_py312/bin/activate
pip install -r scripts/env/kaggle_venv_py312_frozen.txt
```

**Module loads in active use**:
```bash
# GPU work (GH200/aarch64 nodes)
module use /appl/modulefiles/manual/aida/aarch64
module load python-pytorch/2.10

# CPU-only work (x86_64 login/small-partition nodes)
module use /appl/modulefiles/manual/aida/x86_64
module load python-pytorch/2.10
```

**`privatemodules`**: `~/privatemodules/x86_64/` exists but is empty, no
custom modulefiles to carry.

## Home items to carry to the new account, by name

- Dotfiles: `.zshrc`, `.zshenv`, `.p10k.zsh`, `.bashrc`, `.bash_profile`,
  `.gitconfig` (update the email in it first), `.config/`, `.oh-my-zsh/`
- The repo clone itself (`projects/sleepfm-mesa`) — or a fresh `git clone`
  from wherever it's pushed, which is simpler than copying
- `privatemodules/` (currently empty, carry the empty dir or just recreate it)

**Must NOT be copied**:
- **Shell history** (`.bash_history`, `.zsh_history`) — may contain
  tokens or other sensitive strings typed at the command line. Do not
  copy, do not merge into the new account's history.
- **SSH private key** (`.ssh/id_ed25519`) — do not copy a private key
  between accounts. Generate a new keypair on the new account instead and
  add the new public key to GitHub (and anywhere else the old key was
  authorized).

**Git email**: `.gitconfig`'s / the repo's committer email is currently
the literal placeholder `your-email@example.com` — this needs a real
value on the new account before any further commits, it was never
supplied this session.

## NSRR downloader

- Binary: `/users/hamehdi/.local/share/x86_64/gem/ruby/3.4.0/bin/nsrr`
  (a Ruby gem executable, installed under home, not under `miniconda3`).
- Gem environment: `~/.local/share/x86_64/gem/ruby/3.4.0/`.
- Token: read from `.env` at the repo root (`NSRR_TOKEN=...`) by the
  Python download scripts (`download_mesa.py`, `download_mros.py`,
  `download_shhs.py`) — path only, value not reproduced here.
- `/scratch/project_2019517/miniconda3/share/rubygems/cache/nsrr-8.0.0.gem`
  is a cached but apparently unused download — `gem list nsrr` against
  that environment shows nothing installed; the real, working install is
  the home-directory one above.

## Scratch purge policy and what's at risk

`/scratch/project_2019517` has a 180-day cleanup policy (confirmed via
`csc-workspaces`, "Cleanup" column = `180d`). Whether that clock is keyed
to modification time or access time was not re-confirmed this session —
treat 180 days since last write as the conservative assumption. At risk
if the project goes untouched that long: `moody2026_pilot/` (all Moody
checkpoints, embeddings, caches, notes — nothing here is in git),
`sleepfm-data/` (MESA/MrOS/SHHS HDF5, embeddings, checkpoints),
`kaggle_venv_py312`, and `miniconda3`. `sleepfm_venv_arm` is safe, it
lives on `/projappl`, which has no stated purge policy here.

## Billing units and usage

`csc-projects` and `csc-quota` were not reachable this session (both
reported "not accessible" for `project_2019517`). Billing/usage could not
be checked — flag as an open item for whoever has working access.

## Post-migration smoke test checklist

- [ ] `module use /appl/modulefiles/manual/aida/aarch64 && module load python-pytorch/2.10`
- [ ] `source /projappl/project_2019517/sleepfm_venv_arm/bin/activate`
- [ ] Submit a tiny GPU job (e.g. a 1-epoch smoke test on an existing config) on `--partition=gpumedium --gres=gpu:gh200:1`
- [ ] Read one HDF5 file from `mesa/hdf5` or `mesa/hdf5_350_puhti_baseline` and confirm the expected keys (`EEG1`, `EEG2`, `EEG3`, `EKG`)
- [ ] Write a small test file to `/scratch/project_2019517/` and confirm group/permissions land as expected
- [ ] `git pull` on the repo clone, confirm it fetches cleanly
- [ ] Make a trivial commit and `git push`, confirm the new SSH key (not the old private key) authenticates to GitHub

## Open questions

- Whether the 180-day scratch purge is keyed to modification time or access time (not re-confirmed this session).
- Billing units and current usage for `project_2019517` (csc-projects/csc-quota unreachable this session).
- Whether `sleepfm-data/`'s group change to `project_2019517` should happen before or after `hamehd` is added to the project (ordering not specified).
- Real value for the git committer email, still the placeholder.
