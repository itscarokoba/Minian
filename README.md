# Minian batch runner

Runs the [Minian](https://github.com/denisecailab/minian) calcium-imaging pipeline
on every trial of a recording day, reusing the parameters you tuned by hand on trial 1.

## Files
- `Minian.ipynb` – the Minian pipeline notebook. You tune parameters here on trial 1.
- `run_minian_batch.py` – re-runs that notebook, non-interactively, on all remaining trials.

## Setup (once)
1. Install Minian in its own conda environment:
   ```bash
   conda create -y -n minian -c conda-forge minian
   conda activate minian
   pip install nbclient nbformat
   ```
2. Download this repo (green **Code** button → *Download ZIP*, or `git clone <repo URL>`).

## Expected data layout
```
<DATA_ROOT>/            e.g. /Volumes/data/Golden_Lab_Users/<you>/<animal>
  <DAY>/                e.g. 2025_07_21
    <TRIAL>/            named HH_MM_SS, e.g. 11_02_45
      Miniscope/
        Trial_1.avi  (or 0.avi, 1.avi, ...)
```

## Workflow for each recording day
1. **Tune trial 1 by hand.** Open `Minian.ipynb` (with the `minian` environment active),
   edit the paths in the *Set up Initial Basic Parameters* cell (`dpath`, `out_dir`, `intpath`)
   to point at the day's first trial, keep `interactive = True`, and run it top to bottom,
   adjusting parameters until the results look good.
2. **Save the notebook.** The batch script reads your tuned values from it.
3. **Shut down that notebook's kernel** (frees memory).
4. **Edit the CONFIG block** at the top of `run_minian_batch.py`: set `DATA_ROOT` and `DAY`
   (and `SUBSET_MC` if you drew a motion-correction mask).
5. **Run:**
   ```bash
   conda activate minian
   python run_minian_batch.py                       # all trials of DAY except trial 1
   python run_minian_batch.py 2025_07_22            # a different day
   python run_minian_batch.py 2025_07_21 11_32_05   # specific trial(s), e.g. to re-run a failure
   ```

## Output
Results go to `~/minian_results/<animal>/<DAY>/<TRIAL>/`:
- `minian/` – final results (`A`, `C`, `S`, `max_proj`, `motion`, … as `.zarr`)
- `minian_mc.mp4`, `minian.mp4` – motion-correction and result videos
- `<TRIAL>_executed.ipynb` – the notebook as run, with all plots
- `params_used.json` – the exact parameters used

`batch_log.txt` in the day folder records ok/failed/skipped for each trial.
Trials that already have results are skipped; delete a trial's folder to re-run it.
Intermediate files go to `~/minian_scratch/` and are deleted after each successful trial.

Load results in Python:
```python
import xarray as xr
C = xr.open_zarr("~/minian_results/<animal>/<DAY>/<TRIAL>/minian/C.zarr")["C"]
```
