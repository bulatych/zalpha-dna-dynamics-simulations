"""Protein–DNA interaction energy with FoldX (RepairPDB + AnalyseComplex). Run ``python foldx_utils.py`` for all models."""
import logging
import re
import subprocess
from pathlib import Path

import pandas as pd

import config
from common import cached_csv, complex_for_pdb, load_universe, sample_frames, setup_environment

logger = logging.getLogger(__name__)


def extract_frames_to_pdb(universe, frames, output_dir, prefix):
    """Write frame_<i>_<prefix>.pdb files of the protein–DNA complex (chain IDs and residue names fixed for FoldX)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atoms = complex_for_pdb(universe)
    for ts in universe.trajectory[list(frames)]:
        atoms.write(str(output_dir / f"frame_{ts.frame}_{prefix}.pdb"))
    logger.info("Extracted %d frames to %s", len(frames), output_dir)


def _foldx(args, cwd, executable):
    result = subprocess.run([executable, *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("FoldX %s failed (exit %d): %s", " ".join(args), result.returncode, result.stderr.strip()[-500:])
    return result


def run_foldx_pipeline(frames_dir, executable=config.FOLDX_EXECUTABLE):
    """RepairPDB, then AnalyseComplex (DNA mode) for every PDB in ``frames_dir``; stdout saved as *_Repair_AnalyseComplex.txt."""
    frames_dir = Path(frames_dir)
    for pdb in sorted(p for p in frames_dir.glob("*.pdb") if not p.stem.endswith("_Repair")):
        _foldx(["--command=RepairPDB", f"--pdb={pdb.name}"], frames_dir, executable)
    for pdb in sorted(frames_dir.glob("*_Repair.pdb")):
        result = _foldx(["--command=AnalyseComplex", f"--pdb={pdb.name}", "--complexWithDNA=true"], frames_dir, executable)
        (frames_dir / f"{pdb.stem}_AnalyseComplex.txt").write_text(result.stdout)


def parse_foldx_results(frames_dir):
    """Interaction energy (kcal/mol) per frame from the AnalyseComplex outputs."""
    rows = []
    for path in sorted(Path(frames_dir).glob("*_Repair_AnalyseComplex.txt")):
        energy = re.search(r"interaction between.*?Total\s*=\s*([-\d.]+)", path.read_text(), flags=re.S)
        frame = re.search(r"frame_(\d+)", path.name)
        rows.append({"frame": int(frame.group(1)) if frame else None, "InteractionEnergy_kcal": float(energy.group(1)) if energy else None, "file": path.name})
    return pd.DataFrame(rows).sort_values("frame").reset_index(drop=True)


def run_foldx_analysis(model, frames=None, executable=config.FOLDX_EXECUTABLE, force=None):
    """Extract frames, run FoldX and parse the energies of a model; cached in output/csv."""
    def compute():
        universe = load_universe(model)
        frames_dir = config.FOLDX_DIR / model.stem
        extract_frames_to_pdb(universe, sample_frames(len(universe.trajectory)) if frames is None else frames, frames_dir, model.stem)
        run_foldx_pipeline(frames_dir, executable)
        return parse_foldx_results(frames_dir)
    return cached_csv(model.csv("foldx"), compute, force)


if __name__ == "__main__":
    setup_environment()
    for model in config.MODELS.values():
        run_foldx_analysis(model)
