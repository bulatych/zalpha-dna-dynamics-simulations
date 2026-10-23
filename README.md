# ADAR1 Zα – Z-DNA vs B-DNA molecular dynamics analysis

Analysis and figures for MDS of the ADAR1 Zα domain bound to Z-DNA and B-DNA:
RMSD, per-residue RMSF, protein–DNA contacts (salt bridges, H-bonds, stacking, vdW),
contact lifetimes, contact co-occurrence and, optionally, FoldX / PyRosetta interaction energies.

## Repository structure

```
├── Data_analysis_visualisation.ipynb  # entry point: runs the analyses and draws all figures
├── config.py             # paths, models, analysis parameters, plot style  <- edit this
├── common.py             # shared helpers: I/O + CSV caching, figure saving, RMSD/RMSF, PDB export, plotting helpers
├── contact_analysis.py   # contacts, lifetimes, heatmap / Jaccard matrices
├── foldx_utils.py        # FoldX interaction energy (optional)
├── pyrosetta_utils.py    # PyRosetta interface dG (optional)
├── data/                 # input .prmtop + .xtc (not tracked by git)
└── output/{csv,figures}/ # intermediate tables and final figures
```

## Installation

```bash
git clone <repo-url> && cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```


FoldX and PyRosetta are only needed for the interaction-energy section and are installed separately.

## Usage

1. Put `<stem>.prmtop` and `<stem>.xtc` of every model into `data/`
   (trajectories: <DOI / download link>). To analyse your own systems, edit `MODELS` in `config.py`.
2. Run `Data_analysis_visualisation.ipynb` to make plots.

Each analysis writes its result to `output/csv/<stem>_<analysis>.csv` and later runs read
that file instead of the trajectory, so figures can be re-drawn quickly — or from the
provided CSVs alone, without trajectories. Set `FORCE_RECOMPUTE = True` in `config.py`
to recompute everything.

Scripts can be run from the command line :

```bash
python contact_analysis.py   # contacts + lifetimes for all models
python foldx_utils.py        # FoldX energies
python pyrosetta_utils.py    # PyRosetta interface dG
```

## Outputs

| Intermediate CSV (`output/csv/`) | Content |
|---|---|
| `*_RMSD.csv` | complex RMSD per frame |
| `*_RMSF.csv` | per-residue protein RMSF |
| `*_Interaction_Analysis_4A.csv` | every protein–DNA heavy-atom contact per frame, with interaction type and DNA group |
| `*_contact_lifetimes.csv` | continuous atom–atom contact segments (frames) |
| `*_residue_lifetimes.csv` | continuous residue–DNA contact segments (frames) |
| `*_FoldX_interaction_energy.csv`, `*_Rosetta_interface_dG.csv` | interaction energies of sampled frames |

Figures (`output/figures/`): RMSD, RMSF, the contact-heatmap plot and the
contact co-occurrence heatmaps of each model.
