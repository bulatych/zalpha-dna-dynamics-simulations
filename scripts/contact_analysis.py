"""
Protein–DNA contacts (salt bridge / H-bond / stacking / vdW) with conformation-aware DNA atom groups (Z-DNA convex surface vs
B-DNA major groove), atom- and residue-level contact lifetimes, and the matrices used for the heatmaps and the Jaccard analysis.
Run ``python contact_analysis.py`` to (re)generate the CSV files of all models.
"""
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
from MDAnalysis.lib.distances import distance_array
from scipy.spatial.distance import pdist, squareform
from tqdm.auto import tqdm

import config
from common import cached_csv, dna_strand_chains, drop_caps, load_universe, setup_environment, sort_by_residue_number

# DNA atom groups; convex surface of Z-DNA == major groove of B-DNA
SUGAR_PHOSPHATE_ATOMS = {"P", "O5'", "O3'", "C5'", "C4'", "C3'"}
CONVEX_OR_MAJOR_GROOVE_ATOMS = {"DG": {"N7", "C5", "C6", "O6"}, "DC": {"C5", "C6", "N4", "C4"},
                                "DA": {"C8", "N7", "C5", "C6", "N6"}, "DT": {"C5", "C6", "C7", "O4"}}
MINOR_GROOVE_ATOMS = {"DG": {"N1", "C2", "N2", "N3", "C4", "N9"}, "DC": {"N1", "C2", "O2", "N3"},
                      "DA": {"N1", "C2", "N3", "C4", "N9"}, "DT": {"N1", "C2", "O2", "N3"}}

# Interactions
CATION_ATOMS = {("LYS", "NZ"), ("ARG", "NH1"), ("ARG", "NH2")}
STACKING_RESIDUES = {"PHE": "pi-pi", "TYR": "pi-pi", "TRP": "pi-pi", "HIS": "pi-pi", "ARG": "arg-stacking"}
PROTEIN_RING_ATOMS = {"pi-pi": "name CG CD1 CD2 CE1 CE2 CZ ND1 NE2", "arg-stacking": "name CZ NH1 NH2 NE CD"}
BASE_RING_ATOMS = "name C2 C4 C5 C6 C8 N1 N3 N7 N9"


def surface_group(dna_type):
    return "convex_surface" if dna_type.upper() == "Z-DNA" else "major_groove"


def dna_group_order(dna_type):
    return ["backbone_phosphate_anionic", "backbone_sugar_phosphate", surface_group(dna_type), "guanine_C8", "minor_groove",
            "sugar_O4_exposed", "sugar_other"]


def clean_dna_resname(resname):
    return re.sub(r"[^A-Za-z]", "", resname)  # DG5 -> DG


def classify_dna_atom(name, resname, dna_type="Z-DNA"):
    base = clean_dna_resname(resname)
    if name in ("OP1", "OP2"):
        return "backbone_phosphate_anionic"
    if name in SUGAR_PHOSPHATE_ATOMS:
        return "backbone_sugar_phosphate"
    if name == "O4'":
        return "sugar_O4_exposed"
    if name in ("C1'", "C2'"):
        return "sugar_other"
    if base == "DG" and name == "C8":
        return "guanine_C8"
    if name in CONVEX_OR_MAJOR_GROOVE_ATOMS.get(base, ()):
        return surface_group(dna_type)
    if name in MINOR_GROOVE_ATOMS.get(base, ()):
        return "minor_groove"
    return "base_other"


def ring_geometry(positions):
    """Centroid and unit normal of a planar group of atoms (None if < 3 atoms)."""
    if len(positions) < 3:
        return None
    center = positions.mean(axis=0)
    normal = np.linalg.svd(positions - center)[2][2]
    return center, normal / np.linalg.norm(normal)


def compute_contacts(universe, dna_type, cutoff=config.CONTACT_CUTOFF):
    """Heavy-atom protein–DNA contacts in every frame, classified by interaction type and DNA atom group."""
    protein, dna = universe.select_atoms("protein and not name H*"), universe.select_atoms("nucleic and not name H*")
    hbonds = set()
    for frame, donor, _, acceptor, *_ in HydrogenBondAnalysis(universe, between=["protein", "nucleic"]).run().results.hbonds.astype(int):
        hbonds |= {(frame, donor, acceptor), (frame, acceptor, donor)}

    # static per-atom annotations; DNA residues renumbered from 1 within each strand
    chains, counters, dna_resnum = dna_strand_chains(universe), defaultdict(int), {}
    for res in dna.residues:
        counters[chains[res.resindex]] += 1
        dna_resnum[res.resindex] = counters[chains[res.resindex]]
    prot_res = [f"{a.resname}{a.resid}" for a in protein]
    is_cation = [(a.resname, a.name) in CATION_ATOMS for a in protein]
    dna_tag = [f"{chains[a.resindex]}{dna_resnum[a.resindex]}" for a in dna]
    dna_res = [f"{chains[a.resindex]}:{clean_dna_resname(a.resname)}{dna_resnum[a.resindex]}" for a in dna]
    dna_group = [classify_dna_atom(a.name, a.resname, dna_type) for a in dna]
    prot_rings = {r.resindex: (r.atoms.select_atoms(PROTEIN_RING_ATOMS[kind]), kind) for r in protein.residues if (kind := STACKING_RESIDUES.get(r.resname))}
    base_rings = {r.resindex: ring for r in dna.residues if len(ring := r.atoms.select_atoms(BASE_RING_ATOMS)) >= 3}

    rows = []
    for ts in tqdm(universe.trajectory, desc=f"Contacts ({dna_type})"):
        dist = distance_array(protein.positions, dna.positions)
        geometry = {}  # ring geometry of this frame, computed on demand
        for i, j in zip(*np.nonzero(dist < cutoff)):
            p, d = protein[i], dna[j]
            itype, d_rings, angle, mode = "vdw", None, None, None
            if is_cation[i] and d.name in ("OP1", "OP2"):
                itype = "salt_bridge"
            elif (ts.frame, p.index, d.index) in hbonds:
                itype = "hbond"
            elif p.resindex in prot_rings and d.resindex in base_rings:
                p_atoms, kind = prot_rings[p.resindex]
                for key, atoms in ((("p", p.resindex), p_atoms), (("d", d.resindex), base_rings[d.resindex])):
                    if key not in geometry:
                        geometry[key] = ring_geometry(atoms.positions)
                p_geom, d_geom = geometry[("p", p.resindex)], geometry[("d", d.resindex)]
                if p_geom is not None and d_geom is not None:
                    d_rings = np.linalg.norm(p_geom[0] - d_geom[0])
                    angle = np.degrees(np.arccos(np.clip(np.abs(np.dot(p_geom[1], d_geom[1])), -1, 1)))
                    if d_rings < config.STACKING_MAX_CENTER_DIST:
                        mode = "sandwich" if angle < 30 else "T-shaped" if 60 < angle < 120 else None
                        itype = kind if mode else itype
            rows.append((ts.frame, f"{p.resid}_{p.name}_{dna_tag[j]}_{d.name}", prot_res[i], p.name, dna_tag[j].rstrip("0123456789"),
                         dna_res[j], d.name, dna_group[j], itype, dist[i, j], d_rings, angle, mode))
    return pd.DataFrame(rows, columns=["frame", "contact_id", "prot_res", "prot_atom", "dna_chain", "dna_res", "dna_atom", "dna_group",
                                       "interaction", "distance", "d_rings", "ring_angle", "stacking_mode"])


def lifetime_segments(frames):
    """Runs of consecutive frames as (start, end, length)."""
    frames = np.asarray(frames)
    if frames.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(frames) != 1)
    return [(int(frames[s]), int(frames[e]), int(frames[e] - frames[s] + 1)) for s, e in zip(np.r_[0, breaks + 1], np.r_[breaks, frames.size - 1])]


def compute_atom_lifetimes(contacts):
    """Lifetime (frames) of every continuous atom–atom contact."""
    meta_cols = ["prot_res", "prot_atom", "dna_chain", "dna_res", "dna_atom", "dna_group", "interaction"]
    rows = []
    for contact_id, group in contacts.groupby("contact_id"):
        group = group.sort_values("frame")
        meta = group.iloc[0][meta_cols].to_dict()
        rows += [{"contact_id": contact_id, "start_frame": s, "end_frame": e, "lifetime": n, **meta} for s, e, n in lifetime_segments(group["frame"].values)]
    return pd.DataFrame(rows)


def compute_residue_lifetimes(contacts):
    """Lifetime (frames) of continuous contacts of a protein residue with any DNA residue."""
    mode = lambda x: x.mode()[0]  
    per_frame = (contacts.groupby(["prot_res", "frame"])
                 .agg(dna_res=("dna_res", mode), interaction=("interaction", mode), dna_group=("dna_group", mode), distance=("distance", "min"))
                 .reset_index().sort_values(["prot_res", "frame"]))
    rows = []
    for prot_res, group in per_frame.groupby("prot_res"):
        indexed = group.set_index("frame")
        for start, end, length in lifetime_segments(group["frame"].values):
            seg = indexed.loc[start:end]
            rows.append({"prot_res": prot_res, "start_frame": start, "end_frame": end, "lifetime": length, "dna_res": seg["dna_res"].iloc[0],
                         "interaction_mode": mode(seg["interaction"]), "dna_group_mode": mode(seg["dna_group"]), "mean_distance": seg["distance"].mean()})
    return pd.DataFrame(rows)


def run_contact_and_lifetime_analysis(model, force=None):
    """Contacts + atom/residue lifetimes of a model, cached in output/csv (trajectory is loaded only if needed)."""
    contacts = cached_csv(model.csv("contacts"), lambda: compute_contacts(load_universe(model), model.dna_type), force)
    return {"contacts": contacts,
            "atom_lifetimes": cached_csv(model.csv("atom_lifetimes"), lambda: compute_atom_lifetimes(contacts), force),
            "residue_lifetimes": cached_csv(model.csv("residue_lifetimes"), lambda: compute_residue_lifetimes(contacts), force)}


def _group_matrix(df, row, column, values, agg, dna_type):
    grouped = df.groupby([row, column])
    matrix = grouped.size() if agg == "size" else grouped[values].agg(agg)
    return matrix.unstack(fill_value=0).reindex(columns=dna_group_order(dna_type), fill_value=0)


def residue_contact_matrix(contacts, dna_type):
    """Number of atom contacts: protein residue × DNA group."""
    return _group_matrix(drop_caps(contacts), "prot_res", "dna_group", None, "size", dna_type)


def residue_lifetime_matrix(residue_lifetimes, dna_type, factor=config.LIFETIME_FACTOR):
    """Longest residue contact: protein residue × DNA group."""
    matrix = _group_matrix(drop_caps(residue_lifetimes), "prot_res", "dna_group_mode", "lifetime", "max", dna_type)
    return matrix if factor == 1 else matrix * factor


def atom_lifetime_matrix(atom_lifetimes, dna_type, top_n=config.TOP_N_ATOMS, factor=config.LIFETIME_FACTOR):
    """Longest atom contact: protein atom × DNA group, for the ``top_n`` longest-lived protein atoms."""
    df = drop_caps(atom_lifetimes)
    matrix = _group_matrix(df.assign(prot_atom_id=df["prot_res"] + "_" + df["prot_atom"]), "prot_atom_id", "dna_group", "lifetime", "max", dna_type)
    top = matrix.loc[matrix.max(axis=1).nlargest(top_n).index]
    top = top.reindex(sort_by_residue_number(top.index))
    return top if factor == 1 else top * factor


def align_residues(*matrices):
    """Reindex matrices on the union of their residues, sorted by residue number."""
    residues = sort_by_residue_number(set().union(*(m.index for m in matrices)))
    return [m.reindex(residues, fill_value=0) for m in matrices]


def contact_cooccurrence(contacts, residues=config.JACCARD_RESIDUES, min_occupancy=config.JACCARD_MIN_OCCUPANCY,
                         stable_occupancy=config.STABLE_CONTACT_OCCUPANCY):
    """Jaccard similarity of per-frame DNA-contact patterns of residues + residues in contact in >= stable_occupancy of frames."""
    presence = (contacts.loc[contacts["prot_res"].isin(residues), ["frame", "prot_res"]].drop_duplicates().assign(present=1)
                .pivot(index="frame", columns="prot_res", values="present").fillna(0))
    occupancy = presence.mean(axis=0)
    presence = presence[sort_by_residue_number(occupancy[occupancy >= min_occupancy].index)]
    similarity = pd.DataFrame(1 - squareform(pdist(presence.T, metric="jaccard")), index=presence.columns, columns=presence.columns)
    return similarity, occupancy[occupancy >= stable_occupancy].index.tolist()


if __name__ == "__main__":
    setup_environment()
    for model in config.MODELS.values():
        run_contact_and_lifetime_analysis(model)
