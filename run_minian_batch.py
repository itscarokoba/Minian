"""
Batch-run Minian.ipynb on the remaining trials of one recording day.

Workflow per day:
  1. Run trial 1 manually in Minian.ipynb with interactive = True and tune
     parameters. SAVE the notebook when done (the tuned values live in Cell 14
     AND in later cells like `param_denoise['ksize'] = 5`; this script re-runs
     those cells, so every tuned value carries over automatically).
  2. Shut down the trial-1 kernel (frees RAM / dask workers).
  3. Edit the CONFIG block below (DAY, and SUBSET_MC if you drew a motion mask),
     then run from Terminal:

        conda activate minian
        python run_minian_batch.py

     Optional: pass a day and/or specific trials on the command line, e.g.
        python run_minian_batch.py 2025_07_22
        python run_minian_batch.py 2025_07_21 11_06_10 11_10_33

What it does for each trial:
  - copies Minian.ipynb in memory, appends overrides to the parameters cell
    (dpath, intpath, output paths, interactive = False)
  - redirects the two video-writing cells away from the read-only server
  - drops the unguarded VArrayViewer/panel cell, the hard-coded `chk = {...}`
    recovery cell, and every scratch cell after `cluster.close()`
  - executes it in the `minian` kernel, saving the executed notebook, a log,
    and the effective parameters next to the results
  - on error, records it and moves on to the next trial
"""

import json
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# ============================== CONFIG ======================================
# Folder holding one animal's recording days: DATA_ROOT/<DAY>/<TRIAL>/Miniscope/*.avi
DATA_ROOT = Path("/Volumes/data/Golden_Lab_Users/<YOUR_FOLDER>/<ANIMAL>")
DAY = "2025_07_21"

# Notebook whose (tuned) cells are reused for every trial.
TEMPLATE_NB = Path(__file__).resolve().parent / "Minian.ipynb"

# Local scratch + results (lab server is read-only).
INTERMEDIATE_ROOT = Path.home() / "minian_scratch"  # -> minian_intermediate_<TRIAL>
RESULTS_ROOT = Path.home() / "minian_results" / DATA_ROOT.name  # -> <DAY>/<TRIAL>/

# Trial folders look like HH_MM_SS; anything else (e.g. the GCAMP6S_... baseline)
# is ignored. Trial 1 (first in time order) is skipped because it is run by hand.
TRIAL_NAME_RE = re.compile(r"^\d{2}_\d{2}_\d{2}$")
SKIP_FIRST_TRIAL = True

# Video file names to look for in each trial's Miniscope folder, tried in order;
# the first pattern that matches any file is used for that trial. Some trials were
# renamed to Trial_1.avi, others still have the Miniscope default 0.avi, 1.avi, ...
VIDEO_PATTERNS = [r"^Trial_[0-9]+\.avi$", r"^[0-9]+\.avi$"]

# If you drew a motion-correction mask on trial 1, paste the printed `subset_mc`
# here as a Python expression string, e.g. "{'height': slice(10, 190), 'width': slice(5, 195)}".
# Leave None to use the full frame.
SUBSET_MC = None

# Delete the per-trial intermediate folder after a successful run (it can be
# several GB per trial and is fully regenerable). Kept on failure for debugging.
DELETE_INTERMEDIATE_ON_SUCCESS = True

KERNEL_NAME = "python3"  # resolves to the minian env when run with its python
# ============================================================================

PARAMS_CELL_MARKER = "# Set up Initial Basic Parameters"

# Make the env's bin (ffmpeg etc.) visible to the kernel even without `conda activate`.
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")


def find_trials(day_dir: Path, only=None):
    trials = sorted(
        p.name for p in day_dir.iterdir() if p.is_dir() and TRIAL_NAME_RE.match(p.name)
    )
    if only:
        missing = set(only) - set(trials)
        if missing:
            sys.exit(f"Requested trials not found in {day_dir}: {sorted(missing)}")
        return [t for t in trials if t in only]
    return trials[1:] if SKIP_FIRST_TRIAL else trials


def build_notebook(template, dpath, intpath, out_dir, video_pattern=None):
    nb = nbformat.from_dict(json.loads(json.dumps(template)))  # deep copy
    cells = nb.cells

    # Truncate at the cluster.close() cell: everything after is scratch/exploration.
    close_idx = next(
        i for i, c in enumerate(cells)
        if c.cell_type == "code" and "cluster.close()" in c.source
    )
    cells = cells[: close_idx + 1]

    params_idx = next(
        i for i, c in enumerate(cells)
        if c.cell_type == "code" and c.source.lstrip().startswith(PARAMS_CELL_MARKER)
    )

    kept = []
    for i, c in enumerate(cells):
        if c.cell_type != "code":
            continue
        src = c.source
        # Unguarded interactive viewer served through panel.
        if ".servable()" in src or "pn.panel(" in src:
            continue
        # Hard-coded chunk override used only for kernel-restart recovery.
        if re.match(r"\s*chk\s*=\s*\{", src):
            continue
        if i == params_idx:
            src += f"""

# ---- overrides injected by run_minian_batch.py ----
dpath = {str(dpath)!r}
intpath = {str(intpath)!r}
out_dir = {str(out_dir)!r}
minian_ds_path = os.path.join(out_dir, "minian")
param_save_minian["dpath"] = minian_ds_path
interactive = False
subset_mc = {SUBSET_MC if SUBSET_MC else None}
os.environ["MINIAN_INTERMEDIATE"] = intpath
os.makedirs(out_dir, exist_ok=True)
"""
            if video_pattern:
                src += f'param_load_videos["pattern"] = {video_pattern!r}\n'
        # When launched from a VS Code terminal, panel auto-selects "vscode" comms and
        # renders plots as ipywidgets (needs jupyter_bokeh) -> force plain HTML output.
        if "hv.notebook_extension(" in src:
            src += "\nimport panel as pn\npn.config.comms = 'default'\n"
        # Old holoviews grouped opts (plot=dict(...), style=dict(...)) were removed in
        # holoviews >= 1.18; flatten them: dict(**dict(a=1), **dict(cmap=...)).
        src = re.sub(r"\b(plot|style)=dict\(", "**dict(", src)
        # Video outputs default to dpath (read-only server) -> send to out_dir.
        if "write_video(" in src:
            src = src.replace('"minian_mc.mp4", dpath)', '"minian_mc.mp4", out_dir)')
            assert "out_dir" in src, "could not redirect write_video output"
        if "generate_videos(" in src:
            src = src.replace("vpath=dpath", "vpath=out_dir")
            assert "out_dir" in src, "could not redirect generate_videos output"
        c.source = src
        c.outputs = []
        c.execution_count = None
        kept.append(c)

    # Record the parameters actually used (after any tuning cells ran).
    dump = nbformat.v4.new_code_cell(
        "import json as _json\n"
        "_p = {k: v for k, v in globals().items() if k.startswith('param_')}\n"
        "_p['subset_mc'] = subset_mc\n"
        "_p['chk'] = chk\n"
        "with open(os.path.join(out_dir, 'params_used.json'), 'w') as _f:\n"
        "    _json.dump(_p, _f, indent=2, default=repr)\n"
    )
    kept.insert(len(kept) - 1, dump)  # before cluster.close()
    nb.cells = kept
    return nb


def run_trial(template, day, trial):
    dpath = DATA_ROOT / day / trial / "Miniscope"
    intpath = INTERMEDIATE_ROOT / f"minian_intermediate_{trial}"
    out_dir = RESULTS_ROOT / day / trial

    names = [p.name for p in dpath.glob("*.avi")]
    video_pattern = next(
        (pat for pat in VIDEO_PATTERNS if any(re.search(pat, n) for n in names)), None
    )
    if video_pattern is None:
        return "skipped", f"no video matching {VIDEO_PATTERNS} in {dpath} (found: {names})"
    print(f"  videos: {sorted(n for n in names if re.search(video_pattern, n))}", flush=True)
    if (out_dir / "minian" / "C.zarr").exists():
        return "skipped", f"results already exist in {out_dir} (delete to re-run)"

    out_dir.mkdir(parents=True, exist_ok=True)
    # Start from an empty intermediate folder: open_minian(intpath) loads every
    # zarr in it, and a half-written one from an earlier failed run breaks it.
    if intpath.exists():
        shutil.rmtree(intpath)
    nb = build_notebook(template, dpath, intpath, out_dir, video_pattern)
    client = NotebookClient(
        nb, kernel_name=KERNEL_NAME, timeout=None,
        resources={"metadata": {"path": str(out_dir)}},
    )
    n = len(nb.cells)
    t0 = time.time()
    status, msg = "ok", ""
    try:
        with client.setup_kernel():
            for i, cell in enumerate(nb.cells):
                first = cell.source.strip().splitlines()[0][:70] if cell.source.strip() else ""
                print(f"  [{i + 1:3d}/{n}] {first}", flush=True)
                client.execute_cell(cell, i)
                # Errors inside %%time cells are displayed but not reported as
                # failures by the kernel, so check the outputs ourselves.
                for out in cell.outputs:
                    if out.output_type == "error":
                        evalue = " ".join(out.evalue.split())
                        raise CellExecutionError(
                            f"cell {i + 1}: {out.ename}: {evalue}", out.ename, evalue
                        )
    except CellExecutionError as e:
        status, msg = "failed", str(e).strip().splitlines()[-1]
    except Exception as e:  # kernel died, KeyboardInterrupt in kernel, etc.
        status, msg = "failed", f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        nbformat.write(nb, out_dir / f"{trial}_executed.ipynb")

    mins = (time.time() - t0) / 60
    if status == "ok" and DELETE_INTERMEDIATE_ON_SUCCESS:
        shutil.rmtree(intpath, ignore_errors=True)
    return status, f"{mins:.1f} min {msg}".strip()


def main():
    day = sys.argv[1] if len(sys.argv) > 1 else DAY
    only = sys.argv[2:] or None
    day_dir = DATA_ROOT / day
    if not day_dir.is_dir():
        sys.exit(f"{day_dir} not found - is the lab server mounted?")

    template = nbformat.read(TEMPLATE_NB, as_version=4)
    trials = find_trials(day_dir, only)
    (RESULTS_ROOT / day).mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_ROOT.mkdir(parents=True, exist_ok=True)
    # Snapshot the template so later edits to Minian.ipynb don't change this batch.
    shutil.copy(TEMPLATE_NB, RESULTS_ROOT / day / "template_used.ipynb")

    print(f"Day {day}: running {len(trials)} trials: {trials}\n")
    summary = []
    for k, trial in enumerate(trials, 1):
        print(f"=== ({k}/{len(trials)}) {trial} ===", flush=True)
        status, msg = run_trial(template, day, trial)
        print(f"=== {trial}: {status.upper()} {msg}\n", flush=True)
        summary.append((trial, status, msg))
        with open(RESULTS_ROOT / day / "batch_log.txt", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M')}\t{trial}\t{status}\t{msg}\n")

    print("Summary:")
    for trial, status, msg in summary:
        print(f"  {trial}: {status} {msg}")


if __name__ == "__main__":
    main()
