"""Shared helpers: environment, I/O and caching, figure saving, RMSD/RMSF, PDB export and plotting helpers."""
import logging
import re
import textwrap
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from MDAnalysis.analysis import rms

import config

logger = logging.getLogger(__name__)


def setup_environment():
    """Silence noisy warnings/logs, create output folders and apply the plot style."""
    for category in (UserWarning, FutureWarning, RuntimeWarning, DeprecationWarning):
        warnings.simplefilter("ignore", category=category)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for name in ("fontTools", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)
    for directory in (config.CSV_DIR, config.FIGURE_DIR, config.FOLDX_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(config.PLOT_RC)


def load_universe(model):
    missing = [str(p) for p in (model.prmtop, model.xtc) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Input files for {model.key} not found: {', '.join(missing)}")
    return mda.Universe(str(model.prmtop), str(model.xtc), topology_format="PRMTOP")


def cached_csv(path, compute, force=None):
    """Read ``path`` if it exists, otherwise call ``compute()`` and save the result there."""
    path = Path(path)
    if path.exists() and not (config.FORCE_RECOMPUTE if force is None else force):
        logger.info("Reading cached %s", path.name)
        return pd.read_csv(path)
    df = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %s", path)
    return df


def save_figure(fig, name, formats=("png",), dpi=config.FIGURE_DPI, tight=True):
    """Save ``fig`` to output/figures/<name>.<fmt> for every format."""
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(config.FIGURE_DIR / f"{name}.{fmt}", dpi=dpi, bbox_inches="tight" if tight else None)
    print("Saved:", ", ".join(f"{name}.{fmt}" for fmt in formats))


def residue_number(label):
    """'LYS37' / 'ARG42_NH1' -> 37 / 42."""
    match = re.search(r"\d+", str(label))
    return int(match.group()) if match else 0


def sort_by_residue_number(labels):
    return sorted(labels, key=residue_number)


def drop_caps(df, column="prot_res"):
    return df[~df[column].str.startswith(config.CAP_RESNAMES)]


def sample_frames(n_frames, step=config.ENERGY_FRAME_STEP):
    """Every ``step``-th frame plus the last one."""
    return sorted(set(range(0, n_frames, step)) | {n_frames - 1})


def dna_strand_chains(universe):
    """DNA residue index -> chain ID, one chain per covalently bonded strand."""
    strands = universe.select_atoms("nucleic").fragments
    if len(strands) > len(config.DNA_CHAIN_IDS):
        raise ValueError(f"Found {len(strands)} DNA strands; extend config.DNA_CHAIN_IDS.")
    return {res.resindex: chain for chain, strand in zip(config.DNA_CHAIN_IDS, strands) for res in strand.residues}


def compute_rmsd(universe, selection="protein or nucleic", ref_frame=0):
    """Complex RMSD (Å) after superposition on ``ref_frame``; time in ps as reported by MDAnalysis."""
    atoms = universe.select_atoms(selection)
    frame, time, rmsd = rms.RMSD(atoms, atoms, select=selection, ref_frame=ref_frame).run().results.rmsd.T
    return pd.DataFrame({"frame": frame.astype(int), "time_ps": time, "rmsd": rmsd})


def compute_rmsf(universe, resid_range=config.RMSF_RESID_RANGE):
    """Per-residue protein RMSF (Å): mean over the residue's atoms."""
    caps = " ".join(config.CAP_RESNAMES)
    atoms = universe.select_atoms(f"protein and not resname {caps} and resid {resid_range[0]}-{resid_range[1]}")
    per_residue = pd.Series(rms.RMSF(atoms).run().results.rmsf).groupby(atoms.resindices).mean()
    res = atoms.residues
    return pd.DataFrame({"resid": res.resids, "resname": res.resnames, "rmsf": per_residue.loc[res.resindices].to_numpy()})


# PDB export for FoldX/PyRosetta: standard residue names, chain IDs, no caps and hydrogens
AMBER_TO_PDB_RESNAME = {"HIE": "HIS", "HID": "HIS", "HIP": "HIS", "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU", "LYN": "LYS"}


def complex_for_pdb(universe):
    """Protein + DNA heavy atoms ready to be written as PDB (modifies the universe topology in place)."""
    if not hasattr(universe.atoms, "chainIDs"):
        universe.add_TopologyAttr("chainIDs")
    protein = universe.select_atoms("protein")
    protein.chainIDs = config.PROTEIN_CHAIN_ID
    for res in protein.residues:
        res.resname = AMBER_TO_PDB_RESNAME.get(res.resname, res.resname)
    chains = dna_strand_chains(universe)
    for res in universe.select_atoms("nucleic").residues:
        res.atoms.chainIDs = chains[res.resindex]
        res.resname = re.sub(r"[35]$", "", res.resname)  # DG5 / DC3 -> DG / DC
    return universe.select_atoms(f"(protein or nucleic) and not resname {' '.join(config.CAP_RESNAMES)} and not name H*")


# Plotting helpers
PRETTY_GROUP = {"backbone_phosphate_anionic": "backbone phosphate\n(anionic)", "backbone_sugar_phosphate": "backbone\nsugar-phosphate",
                "convex_surface": "convex surface (Z) /\nmajor groove (B)", "major_groove": "convex surface (Z) /\nmajor groove (B)",
                "guanine_C8": "guanine C8", "minor_groove": "minor groove", "sugar_O4_exposed": "sugar O4'\n(exposed)", "sugar_other": "sugar C1', C2'"}
DNA_GROUP_LEGEND_TITLE = "Atoms of DNA Groups (DG = Guanine, DC = Cytosine)"
DNA_GROUP_LEGEND_COLUMNS = (
    "• Backbone Phosphate (anionic): OP1, OP2\n• Backbone Sugar-Phosphate: P, O5', O3', C5', C4', C3'\n"
    "• Convex Surface (Z) / Major Groove (B):\n   DG: N7, C5, C6, O6  |  DC: C5, C6, N4, C4",
    "• Guanine C8: DG: C8\n• Minor Groove: DG: N1, C2, N2, N3, C4, N9  |  DC: N1, C2, O2, N3\n"
    "• Sugar O4' (exposed): O4'\n• Sugar C1', C2': C1', C2'",
)


def pretty_group_label(group):
    return PRETTY_GROUP.get(group, "\n".join(textwrap.wrap(group.replace("_", " "), 16)))


def highlight_tick_labels(labels, highlighted, color=None, fontsize=None):
    """Bold (and optionally recolour / resize) tick labels whose text is in ``highlighted``."""
    for tick in labels:
        if tick.get_text() in highlighted:
            tick.set_fontweight("bold")
            if color:
                tick.set_color(color)
        if fontsize:
            tick.set_fontsize(fontsize)


def heatmap_annotations(data, is_float):
    """Cell labels: integers, or floats with one decimal below 100."""
    fmt = (lambda v: f"{v:.0f}" if (v >= 100 or v % 1 == 0) else f"{v:.1f}") if is_float else (lambda v: f"{int(v)}")
    return np.vectorize(fmt, otypes=[object])(data.to_numpy())


def draw_heatmap(ax, data, cmap, norm, is_float, ylabel, show_xlabels, fs_y, fs_annot, panel_height, fs_axes=8, fs_xticks=7):
    """One heatmap panel of the A4 page; y labels are thinned out when rows are too dense."""
    n_rows, n_cols = data.shape
    label_step = int(np.ceil(fs_y * 1.1 / (panel_height * 72 / n_rows)))
    sns.heatmap(data, ax=ax, annot=heatmap_annotations(data, is_float), fmt="", cmap=cmap, norm=norm, cbar=False,
                linewidths=0.3, linecolor="whitesmoke", yticklabels=True, annot_kws={"size": fs_annot})
    ax.set_yticklabels([t.get_text() if i % label_step == 0 else "" for i, t in enumerate(ax.get_yticklabels())], rotation=0, fontsize=fs_y)
    ax.tick_params(axis="y", pad=2, length=2)
    ax.set_ylabel(ylabel, fontsize=fs_axes, labelpad=3)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    if show_xlabels:
        ax.set_xticklabels([pretty_group_label(c) for c in data.columns], rotation=90, ha="center", va="top", fontsize=fs_xticks, linespacing=1.0)
        ax.tick_params(axis="x", pad=2, length=2)
        ax.set_xlabel("DNA group", fontsize=fs_axes, labelpad=4)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=2)
        ax.set_xlabel("")


def draw_colorbar(fig, cax, cmap, norm, label, integer, fontsize):
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=fontsize, length=2, pad=1.5)
    cb.locator = MaxNLocator(nbins=4, integer=integer)
    cb.update_ticks()
    cb.set_label(label, fontsize=fontsize + 1, labelpad=2)


def draw_dna_group_legend(ax):
    """Rounded footer box listing the atoms of every DNA group."""
    ax.axis("off")
    ax.add_patch(mpatches.FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.03", facecolor="#F8F9FA",
                                         edgecolor="#D1D5DB", linewidth=0.8, transform=ax.transAxes, clip_on=False))
    ax.text(0.02, 0.88, DNA_GROUP_LEGEND_TITLE, fontsize=7.5, color="#1F2937", transform=ax.transAxes, va="top")
    for x, text in zip((0.02, 0.52), DNA_GROUP_LEGEND_COLUMNS):
        ax.text(x, 0.68, text, fontsize=6.5, color="#374151", transform=ax.transAxes, va="top", linespacing=1.35)


def plot_jaccard_heatmap(similarity, stable_residues, dna_label, color=config.STABLE_HIGHLIGHT_COLOR, stable_occupancy=config.STABLE_CONTACT_OCCUPANCY):
    """Jaccard similarity matrix of residue contacts; stable residues are outlined and highlighted."""
    fig = plt.figure(figsize=(10, 9))
    ax = sns.heatmap(similarity, annot=similarity.map(lambda x: f"{x:.2f}"), fmt="", cmap="YlGnBu", vmin=0, vmax=1, square=True,
                     linewidths=0.75, linecolor="white", annot_kws={"size": 11, "weight": "normal"},
                     cbar_kws={"shrink": 0.8, "label": "Jaccard Similarity Index"})
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(12)
    cbar.ax.tick_params(labelsize=11)
    for labels in (ax.get_xticklabels(), ax.get_yticklabels()):
        highlight_tick_labels(labels, stable_residues, color=color, fontsize=11)
    order = list(similarity.columns)
    for res in (r for r in stable_residues if r in order):
        ax.add_patch(plt.Rectangle((order.index(res), order.index(res)), 1, 1, fill=False, edgecolor=color, lw=2))
    ax.set_title(f"Co-occurrence of ADAR1-Zα contacts with {dna_label}", fontsize=15, pad=15)
    ax.set_xlabel("Protein residue", fontsize=13, labelpad=10)
    ax.set_ylabel("Protein residue", fontsize=13, labelpad=10)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    patch = mpatches.Patch(color=color, label=f"Stable contact (≥{int(stable_occupancy * 100)}% occupancy)")
    plt.legend(handles=[patch], loc="lower center", bbox_to_anchor=(0.5, -0.22), frameon=True, fontsize=11)
    plt.tight_layout()
    return fig
