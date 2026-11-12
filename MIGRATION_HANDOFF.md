# Migration Handoff

Written 2026-11-11. Nothing in this document was run as a live command beyond
what was needed to gather the facts below. No secrets, keys, or token
contents appear anywhere in this file, by path only where relevant.

## Post-migration status (11 Nov 2026)

The account migration to **`hamehd`** (new home `/users/hamehd`, old home
`/users/hamehdi` now fully permission-denied) is functionally complete.
Summary below; see the migration verification sessions for full evidence.

**Checks that passed**: identity/groups/quotas, all listed project data
paths (group already corrected to `project_2019517` for both
`sleepfm-data` and `moody2026_pilot`), MESA/Moody file counts (2056/2056,
350/350, 1135/1135, all exact), archived-labels checksum integrity, 40
random HDF5 opens (20 MESA + 20 Moody), the GPU environment smoke test
(GH200, CUDA, all required imports, 0 package-version mismatches vs.
`scripts/env/sleepfm_venv_arm_frozen.txt`), repo state (`results/` holds
exactly the 8 `.md` files; `HEAD` clean), `git config user.name`/`user.email`
now real values (`Hassan Mehdi` / `itshmehdi@gmail.com`, no longer the
placeholder), and a `grep -rn hamehdi` across every tracked script/config/
template/dotfile coming back clean.

**The stale-path bug, found and fixed.** 299 saved `config.json` files
under `data/checkpoints/full_cohort/*/fold_*/` (every full-cohort
fine-tuning run) hardcode the old account's home directory in one or more
of `results_dir` (297 files), `split_path` (239), `labels_path` (174),
`model_path` (40), and `data_path` (4) — baked in at training time, so an
account migration leaves them pointing at a directory that no longer
exists for this account. Added `rebase_old_home_paths()` to
`sleepfm/utils.py` (recursively rewrites any string starting with
`/users/hamehdi/` to the current `~`, no-op otherwise, nested dicts/lists
included) and called it right after every saved-config load in
`evaluate_sleep_staging.py`, `finetune_sleep_staging.py`'s
`--checkpoint_path` resume branch, `generate_embeddings.py`,
`evaluate_biot.py`, `evaluate_labram.py`, and `evaluate_sensorlm.py`.
`finetune_biot.py`/`finetune_labram.py`/`finetune_sensorlm.py`'s
`--checkpoint_dir` resume path never reads a saved config for paths at all
(experiment identity comes from the directory name, weights from
`latest.pth`/`best.pth`), so there was no call site to attach it to there.
Verified end-to-end straight from the original, unmodified checkpoints
(symlinked into a throwaway dir so writes never touched the real
checkpoint folder): From-Scratch and Next-Token EEG_ONLY fold 0 both
reproduce their saved macro F1/accuracy to 6 decimals, and the real
`test_all_*.pickle` files in the checkpoint folders were confirmed
byte-identical (md5) before and after. Added a unit test,
`tests/test_rebase_old_home_paths.py`.

**Correction (12 Nov 2026)**: the earlier note below mischaracterized this
as something that changed because of the `mesa/hdf5` symlink work. It
didn't. `evaluate_biot.py`/`finetune_biot.py` (and the LaBraM/SensorLM
equivalents) have always defaulted `--hdf5_dir` to the relative path
`data/mesa/hdf5`, and their own `--help` text has always documented that
default as "the built-in 350-subject Puhti-era path" -- by original
design, not a regression. The symlink swap replaced a real 350-subject
copy at that path with a symlink to a byte-identical 350-subject copy
(`hdf5_350_puhti_baseline`); the default's *meaning* never changed.
Every existing SLURM template already passes `--hdf5_dir
.../mesa/hdf5_full` explicitly for full-cohort runs, so production was
never at risk. What omitting the flag actually risked -- running on the
wrong, much smaller cohort with no error at all -- is exactly what the
new guard below now catches (confirmed directly: omitting `--hdf5_dir`
on a BIOT re-evaluation now raises immediately, naming "expected 411
subjects/files from the split, only 66 were actually loaded"). Original
finding, for the record: 83,005 test epochs and macro F1 0.7542 on the
350-subject set vs. 515,461 epochs and macro F1 0.7551 (matching the
saved benchmark exactly) on the full cohort.

**The silently-dropped-subjects guard (12 Nov 2026)**. Every Dataset
loader here (`SleepEventClassificationDataset`/`SetTransformerDataset` in
`sleepfm/models/dataset.py`, `BIOTSleepDataset`, `LaBraMSleepDataset`,
`SensorLMSleepDataset`) silently `continue`d past a subject/file it
couldn't open or match, with no count check against what the split
actually listed -- the wrong-`--hdf5_dir` case above is one way to
trigger it, but a partially-synced HDF5 directory or a stale split file
would too, with the same silent symptom (fewer subjects than intended,
no error). Added `check_subject_count()` to `sleepfm/utils.py`: compares
subjects/files actually loaded against the split's own listed count for
that split name and raises (naming both counts, the shortfall, and the
HDF5/labels directories used) unless `--allow_missing_subjects` is
passed. The per-epoch unscored-label exclusion is a separate, deliberate
filter inside each dataset's per-epoch loop and is never counted as
"missing" by this check. Wired into every loader in the paragraph above
plus `generate_embeddings.py`. Along the way, fixed a real, independent
bug the new import exposed: `biot_dataset.py`/`labram_dataset.py` (and,
defensively, `sensorlm_dataset.py`) did a bare `from utils import ...`,
which is ambiguous whenever BIOT/LaBraM's own unrelated `utils.py` is
also on `sys.path` -- whichever module another import happened to cache
first under the name `utils` silently won. Now loaded from its exact
file path via `importlib`, immune to import order. Verified on the real
checkpoints (no copies, no patches, temp output folders so nothing wrote
into a checkpoint folder): BIOT EEG_ONLY fold 0 with the wrong
`--hdf5_dir` now fails loudly with the exact counts; with the correct
one it still gives macro F1 0.7551 / accuracy 0.8311 exactly; From-Scratch
EEG_ONLY fold 0 on MESA and the Moody loader dry run both still pass with
identical counts to before (1135 subjects, 956,610 epochs); the real
`test_all_*.pickle` files were confirmed byte-identical (md5) throughout.
Unit test: `tests/test_check_subject_count.py`.

**Raw predictions archived (12 Nov 2026)**. The 1420 MESA and 800 Moody
raw-prediction pickle files under `checkpoints/full_cohort` and
`moody2026_pilot` (originals left in place, nothing deleted) are now also
archived compressed:
`/projappl/project_2019517/archive/raw_predictions_mesa_2026-11-12.tar.zst`
(5.4 GiB) and `raw_predictions_moody_2026-11-12.tar.zst` (1.2 GiB), each
with a `.md5` sidecar, mode 640, group `project_2019517`. zstd was chosen
over gzip after measuring both on 30 random pickles (near-identical
compression ratio, ~12x faster). To extract:
```bash
tar --use-compress-program="zstd -d" -xf raw_predictions_mesa_2026-11-12.tar.zst -C <dest>
tar --use-compress-program="zstd -d" -xf raw_predictions_moody_2026-11-12.tar.zst -C <dest>
```
The MESA archive's paths are relative to the repo root; the Moody
archive's paths are relative to `/scratch/project_2019517/moody2026_pilot/`.
Verify with `md5sum -c <name>.tar.zst.md5`.

**Scratch purge, re-confirmed (12 Nov 2026)**: per CSC's own docs
(docs.csc.fi), files on `/scratch` are deleted after **180 days without
being accessed** -- access time, not modification time. A file that's
never opened again, even if nothing about it changes, is still on the
clock.

**Open fixes** (commands in the relevant section below, none run):
1. **GitHub SSH key** — a new ed25519 key was generated on this account
   (`SHA256:nqAeAjVmy7vqboJYncqdjkMSHUTHxnHQ5KzG/xdXXlk`) but its public
   half isn't added to GitHub yet; `git fetch`/`push`/`ssh -T` all fail
   with `Permission denied (publickey)`.
2. **NSRR gem** — binary and Ruby gem environment not installed on this
   account yet (`~/.local/share/x86_64/gem/ruby/3.4.0/bin/nsrr` missing).
3. **Kaggle token** — no credentials at either `~/.kaggle/access_token` or
   `~/.config/kaggle/kaggle.json`.
4. **Billing visibility** — `csc-projects`/`csc-quota` still report
   `project_2019517` "not accessible" on this account, same as the old one;
   GPU billing-unit balance remains unconfirmed.

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
