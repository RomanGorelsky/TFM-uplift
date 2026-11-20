# TFM-uplift: foundation models for uplift recommendation

Code for the comparison study. The first piece is a pilot on the Hillstrom e-mail RCT
(women's e-mail vs no e-mail, outcome = visit) comparing TabPFN T/S-learners, CausalPFN and
LightGBM T/S-learners at 2k, 8k and 25k training rows × 3 seeds, scored with Qini, AUUC and
uplift@k. Every method has the shape `tau_hat = method(y, T, X, X_test)`, as in Zhang et al.'s
`section3_2_CATE/`.

## Layout

| Path | What it is |
|---|---|
| `pilot/data.py` | Hillstrom loading (scikit-uplift), two arms, one-hot encoding, stratified splits |
| `pilot/methods.py` | the methods, all with one signature, in `REGISTRY` |
| `pilot/run_pilot.py` | method × size × seed loop; resumable; writes `results/<name>/results.csv` |
| `pilot/summarize.py` | `summary.md`, `qini_vs_size.png`, `qini_curves.png` |
| `slurm/env.sh` | venv, cache folders under `~/TFM-uplift/models`, data under `~/TFM-uplift/data`, TabPFN key |
| `slurm/gpu_test.sbatch` | 30-minute check: GPU, key, one small run |
| `slurm/pilot.sbatch` | the full pilot on one H100 |

`results/`, `data/` and `models/` are not in git; results come back to the Mac through the S3 mirror.

## One-time setup

TabPFN needs a free licence key: log in at Prior Labs, accept the licence and copy the API key.

Cluster:

```bash
cd ~
git clone https://github.com/RomanGorelsky/TFM-uplift.git
cd TFM-uplift
export UV_NO_CACHE=1 PIP_NO_CACHE_DIR=1
uv venv ~/venvs/uplift --python 3.12
source ~/venvs/uplift/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -r requirements.txt
printf '%s\n' '<your TabPFN API key>' > ~/.tabpfn_token
chmod 600 ~/.tabpfn_token
```

## Run

Cluster, first the 30-minute check, then the pilot:

```bash
cd ~/TFM-uplift
git pull
sbatch slurm/gpu_test.sbatch
squeue --me
```

```bash
tail -n 40 slurm/logs/gputest_<jobid>.out
sbatch slurm/pilot.sbatch
```

The pilot is resumable: if it stops, `sbatch slurm/pilot.sbatch` again and it skips finished runs.

Mac, results back:

```bash
aws s3 sync s3://shared-cluster-november/homes/Roman.Gorelskii@mbzuai.ac.ae/TFM-uplift/results/pilot results/pilot --region ap-northeast-1
```

Offline smoke test (any machine, no GPU or downloads):

```bash
python pilot/run_pilot.py --synthetic --methods random lgbm_t lgbm_s --device cpu --out results/smoke
```

## Notes

- `tabpfn` 10.2.0 uses model v3.5 by default; `export TABPFN_MODEL_VERSION=v2` in `slurm/env.sh`
  matches Zhang et al. `results/<name>/meta.json` records package versions.
- CausalPFN reads 4,096 rows by default and retrieves neighbours beyond that; try
  `--causalpfn-ctx 8192` if memory allows.
- If `torch.cuda.is_available()` is false in the GPU check, the CUDA wheel doesn't match the
  driver: reinstall torch with the index URL for the driver's CUDA version (see `nvidia-smi`).
- Baselines are untuned and there is one dataset: a feasibility check, not a paper result.
