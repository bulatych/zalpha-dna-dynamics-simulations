"""Paths, simulated models, analysis parameters and plot style. Edit this file to run the pipeline on your own trajectories."""
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR, OUTPUT_DIR = ROOT_DIR / "data", ROOT_DIR / "output"   # inputs: data/<stem>.prmtop + data/<stem>.xtc
CSV_DIR, FIGURE_DIR, FOLDX_DIR = OUTPUT_DIR / "csv", OUTPUT_DIR / "figures", OUTPUT_DIR / "foldx"
FORCE_RECOMPUTE = False  # True: recompute intermediate CSVs even if they already exist

# Analysis parameters
CONTACT_CUTOFF = 4.0             # Å, heavy-atom protein–DNA contacts
STACKING_MAX_CENTER_DIST = 5.0   # Å, max ring-centroid distance for stacking
CAP_RESNAMES = ("ACE", "NME")
PROTEIN_CHAIN_ID, DNA_CHAIN_IDS = "A", ("B", "C")  # chain IDs in PDBs for FoldX/PyRosetta 
RMSF_RESID_RANGE = (1, 70)
RESIDUE_NUMBERING_OFFSET = 132   # structure numbering -> full-length ADAR1 numbering
LITERATURE_RESIDUES = {37: 169, 38: 170, 41: 173, 42: 174, 45: 177, 60: 192, 61: 193, 63: 195}
TOP_N_ATOMS = 20
JACCARD_RESIDUES = ["SER2", "ILE3", "TYR4", "HIE27", "PRO36", "LYS37", "LYS38", "GLU39", "ASN41", "ARG42",
                    "TYR45", "SER46", "LYS49", "LYS55", "GLY58", "THR59", "PRO60", "PRO61", "TRP63"]
JACCARD_MIN_OCCUPANCY = 0.0
STABLE_CONTACT_OCCUPANCY = 0.98
ENERGY_FRAME_STEP = 50           # FoldX/PyRosetta: every N-th frame + the last one
FOLDX_EXECUTABLE = "foldx"
# MDAnalysis time is in ps, lifetimes are in frames; 1.0 keeps the figures as published
RMSD_TIME_FACTOR = 1.0
LIFETIME_FACTOR = 1.0

CSV_SUFFIXES = {"rmsd": "RMSD", "rmsf": "RMSF", "contacts": f"Interaction_Analysis_{CONTACT_CUTOFF:g}A",
                "atom_lifetimes": "contact_lifetimes", "residue_lifetimes": "residue_lifetimes",
                "foldx": "FoldX_interaction_energy", "rosetta": "Rosetta_interface_dG"}


@dataclass(frozen=True)
class Model:
    key: str          # short ID ("Z-DNA"), also used in figure titles
    stem: str         # file stem of data/<stem>.prmtop and data/<stem>.xtc
    dna_type: str     # "Z-DNA" or "B-DNA": convex surface vs major groove labels
    label: str        # RMSD legend
    short_label: str  # RMSF legend
    color: str

    prmtop = property(lambda self: DATA_DIR / f"{self.stem}.prmtop")
    xtc = property(lambda self: DATA_DIR / f"{self.stem}.xtc")

    def csv(self, kind):
        return CSV_DIR / f"{self.stem}_{CSV_SUFFIXES[kind]}.csv"


MODELS = {
    "Z-DNA": Model("Z-DNA", "ZDNA_ADAR1_Zalpha", "Z-DNA", "Z-DNA / ADAR1 Zα", "Z-DNA model", "#ff7f0e"),
    "B-DNA": Model("B-DNA", "BDNA_ADAR1_Zalpha", "B-DNA", "B-DNA / ADAR1 Zα", "B-DNA model", "#1f77b4"),
}

# Plot style
PLOT_RC = {"text.antialiased": True, "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
           "pdf.fonttype": 42, "ps.fonttype": 42}
FIGURE_DPI = 300
STABLE_HIGHLIGHT_COLOR = "mediumblue"
