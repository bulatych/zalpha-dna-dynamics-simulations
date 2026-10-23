"""Protein–DNA interface energy with PyRosetta InterfaceAnalyzerMover. PyRosetta is imported lazily. Run ``python pyrosetta_utils.py`` for all models."""
import tempfile
from pathlib import Path

import pandas as pd

import config
from common import cached_csv, complex_for_pdb, load_universe, sample_frames, setup_environment

INTERFACE = f"{config.PROTEIN_CHAIN_ID}_{''.join(config.DNA_CHAIN_IDS)}"  # protein vs both DNA strands: "A_BC"
_score_function = None


def get_score_function():
    """Initialise PyRosetta once per session and return the default score function."""
    global _score_function
    if _score_function is None:
        import pyrosetta
        pyrosetta.init(extra_options="-ignore_unrecognized_res -ignore_zero_occupancy false", silent=True)
        _score_function = pyrosetta.get_score_function()
    return _score_function


def compute_interface_energies(model, frames=None, interface=INTERFACE):
    """Interface dG (REU), buried SASA (Å²) and dG/dSASA for the selected frames of a model."""
    import pyrosetta
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    sfxn, universe = get_score_function(), load_universe(model)
    atoms = complex_for_pdb(universe)
    frames = sample_frames(len(universe.trajectory)) if frames is None else frames
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        pdb_path = str(Path(tmp) / "frame.pdb")
        for ts in universe.trajectory[list(frames)]:
            atoms.write(pdb_path)
            iam = InterfaceAnalyzerMover(interface, True, sfxn)
            iam.apply(pyrosetta.pose_from_pdb(pdb_path))
            dg, sasa = iam.get_interface_dG(), iam.get_interface_delta_sasa()
            rows.append({"frame": ts.frame, "ΔG_interface_REU": dg, "dSASA_A2": sasa, "ΔG_normalized": dg / sasa if sasa > 0 else 0.0})
    return pd.DataFrame(rows)


def run_interface_analysis(model, frames=None, force=None):
    """PyRosetta interface energies of a model, cached in output/csv."""
    return cached_csv(model.csv("rosetta"), lambda: compute_interface_energies(model, frames), force)


if __name__ == "__main__":
    setup_environment()
    for model in config.MODELS.values():
        run_interface_analysis(model)
