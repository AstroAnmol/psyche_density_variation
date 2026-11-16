"""
bed_shaking_analysis.py
=======================
Analysis of the top layer of the granular bed before, during, and after
shaking in the simulations under `bed_shaking/`.

The top layer is defined as 2 particle diameters (2 * d) measured downwards
from the local or global bed surface.

Key Capabilities
----------------
1. Automated parameter extraction:
   Reads particle diameter (d), timestep (dt), shaking duration, settling duration,
   and material properties (Kamacite, Troilite, Rocky) from simulation scripts
   (in.shaking, in.bed_preparation, log.*, output*.txt).

2. Top-Layer Identification (2 Particle Diameters):
   - 'grid' (Default): Subdivides the horizontal (x, y) domain into a 2D grid
     (cell size ~ 2-3 d), finds the local surface height z_surf(x, y), and selects
     particles satisfying z >= z_surf(x, y) - 2*d. Automatically filters out transient
     airborne / splashed particles during shaking.
   - 'global_percentile': Uses high percentile of particle elevation (e.g. 99th percentile)
     to determine the surface height and isolates the top 2*d layer.
   - 'global_max': Uses max(z) - 2*d.

3. Multi-Phase Analysis:
   - Before Shaking: Initial settled state (t <= t_shake_start).
   - During Shaking: Active floor oscillation phase (t_shake_start < t <= t_shake_end).
   - After Shaking: Post-vibration settling phase (t > t_shake_end).

4. Metrics Computed:
   - Number count and fraction (%) of each particle type in the top layer.
   - Mass and mass fraction (%) of each particle type in the top layer.
   - Segregation ratio / enrichment factor: S_i = (Pct_i in top layer) / (Pct_i in bulk bed).
   - Bed surface height and bed dilation / expansion during shaking vs compaction after.
   - Surface roughness (RMS deviation of surface elevation).
   - Granular temperature and mean speed in the top layer.

5. Visualizations & Exports:
   - Composition time series with shaded phase bands (Before, During, After).
   - Grouped bar chart comparing composition across phases.
   - Surface elevation dilation and granular temperature time series.
   - 2D spatial distribution / scatter maps of the top layer at key snapshots.
   - CSV outputs: top_layer_time_series.csv and top_layer_phase_summary.csv.
   - Text summary report.

Usage
-----
# Single simulation directory:
python analysis/bed_shaking_analysis.py --sim_dir bed_shaking/seed_1

# All seeds under bed_shaking:
python analysis/bed_shaking_analysis.py --base_dir bed_shaking

# Run built-in self-test / demo with synthetic data:
python analysis/bed_shaking_analysis.py --demo
"""

import os
import re
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

# Set MPLCONFIGDIR to /tmp before importing matplotlib
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp"

import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend safe for headless servers / batch jobs
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Import shared utilities from the analysis package if available
# ---------------------------------------------------------------------------
try:
    from physical_params import read_single_dump, get_dump_file_header
    from particle_distribution import extract_particle_properties, format_youngs_modulus
except ImportError:
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from physical_params import read_single_dump, get_dump_file_header
        from particle_distribution import extract_particle_properties, format_youngs_modulus
    except ImportError:
        read_single_dump = None
        get_dump_file_header = None
        extract_particle_properties = None
        format_youngs_modulus = None


# Default regolith particle type mapping and styling
DEFAULT_TYPE_PROPERTIES = {
    1: {"name": "Floor", "material": "Kamacite", "density": 7870.0, "color": "#708090"},
    2: {"name": "Kamacite", "material": "Fe-Ni alloy", "density": 7870.0, "color": "#2b5c8f"},
    3: {"name": "Troilite", "material": "FeS", "density": 4900.0, "color": "#d9822b"},
    4: {"name": "Rocky", "material": "Silicate regolith", "density": 2500.0, "color": "#c0392b"},
    5: {"name": "Wall", "material": "Acrylic", "density": 1250.0, "color": "#7f8c8d"},
}


# ---------------------------------------------------------------------------
# Fallback dump reader if physical_params is unavailable
# ---------------------------------------------------------------------------
def _fallback_read_single_dump(file_path: Union[str, Path]) -> Tuple[dict, pd.DataFrame]:
    """Read a custom LAMMPS/LIGGGHTS dump file."""
    file_path = Path(file_path)
    metadata = {}
    header_cols = None

    with open(file_path, "r", errors="ignore") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line_str = line.strip()
            if line_str.startswith("ITEM: TIMESTEP"):
                metadata["timestep"] = int(f.readline().strip())
            elif line_str.startswith("ITEM: NUMBER OF ATOMS"):
                metadata["num_atoms"] = int(f.readline().strip())
            elif line_str.startswith("ITEM: BOX BOUNDS"):
                bounds = []
                for _ in range(3):
                    bounds.append([float(x) for x in f.readline().split()])
                metadata["box_bounds"] = np.array(bounds)
            elif line_str.startswith("ITEM: ATOMS"):
                header_cols = line_str.replace("ITEM: ATOMS", "").strip().split()
                break

    if header_cols is None:
        raise ValueError(f"Could not find 'ITEM: ATOMS' line in dump file: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        skiprows=9,
        names=header_cols,
        engine="c",
    )
    return metadata, df


def safe_read_dump(file_path: Union[str, Path]) -> Tuple[dict, pd.DataFrame]:
    """Unified wrapper to read single dump file."""
    if read_single_dump is not None:
        try:
            return read_single_dump(file_path)
        except Exception:
            return _fallback_read_single_dump(file_path)
    return _fallback_read_single_dump(file_path)


# ---------------------------------------------------------------------------
# Parameter Extraction from Simulation Scripts
# ---------------------------------------------------------------------------
def _extract_step_var_seconds(text: str) -> Dict[str, float]:
    """Extract LAMMPS step-count variables defined as `<seconds>/${dt}` or `<seconds>/dt`."""
    var_secs = {
        "two_second_steps": 2.0,
        "one_second_steps": 1.0,
        "three_second_steps": 3.0,
        "five_second_steps": 5.0,
        "half_second_steps": 0.5,
        "0_1_second_steps": 0.1,
        "one_tenth_second_steps": 0.1,
    }
    for m in re.finditer(
        r"^\s*variable\s+(\S+)\s+equal\s+([\d\.]+)/\$\{?dt\}?",
        text,
        re.MULTILINE,
    ):
        try:
            var_secs[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return var_secs


def _parse_run_steps_seconds(text_block: str, dt: float, var_secs: Optional[Dict[str, float]] = None) -> float:
    """Sum physical seconds across non-commented `run <steps>` statements in a script block."""
    if var_secs is None:
        var_secs = _extract_step_var_seconds(text_block)

    total_sec = 0.0
    for line in text_block.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        m = re.match(r"^run\s+(\S+)", stripped)
        if not m:
            continue
        arg = m.group(1)
        var_name = arg.strip("${}")
        if var_name in var_secs:
            total_sec += var_secs[var_name]
        elif re.match(r"^[\d\.]+$", arg):
            total_sec += float(arg) * dt
    return total_sec


def _discover_shaking_scripts(sim_dir: Path) -> List[Path]:
    """
    Discover ordered shaking input scripts (`in.shaking_1`, `in.shaking_2`, ...
    or `in.shaking`) in `sim_dir` (or `sim_dir.parent` if none in `sim_dir`).
    """
    for d in (sim_dir, sim_dir.parent):
        if not d.is_dir():
            continue
        numbered = [
            p for p in d.glob("in.shaking_*")
            if p.is_file() and not p.name.startswith("._")
        ]
        if numbered:
            def _script_order(p: Path) -> Tuple[int, str]:
                m = re.search(r"(\d+)$", p.name)
                return (int(m.group(1)) if m else 0, p.name)
            return sorted(numbered, key=_script_order)
        single = d / "in.shaking"
        if single.is_file():
            return [single]
    return []


def parse_shaking_simulation_parameters(sim_dir: Union[str, Path]) -> dict:
    """
    Parse simulation input scripts (in.shaking / in.shaking_1, in.shaking_2, ...,
    in.bed_preparation, log files) to extract physical parameters, particle properties,
    and per-shaking cycle durations.
    """
    sim_dir = Path(sim_dir)
    search_dirs = [sim_dir, sim_dir.parent]

    search_files = []
    for d in search_dirs:
        if d.is_dir():
            search_files.extend(list(d.glob("in.shaking")))
            search_files.extend(list(d.glob("in.continue")))
            search_files.extend(list(d.glob("in.bed_preparation")))
            search_files.extend(list(d.glob("in.*")))
            search_files.extend(list(d.glob("log.*")))
            search_files.extend(list(d.glob("output*.txt")))

    # Deduplicate
    unique_files = []
    for f in search_files:
        if f.is_file() and f not in unique_files:
            unique_files.append(f)

    # Defaults based on in.shaking
    params = {
        "particle_dia": 0.01,          # 1 cm = 0.01 m
        "particle_radius": 0.005,      # 0.005 m
        "dt": 4e-6,                    # 4 microseconds
        "dumptime": 0.02,              # 20 ms
        "dumpstep": 5000,
        "shaking_amp": 0.001,          # 1 mm
        "shaking_freq": 5.0,           # 5 Hz
        "shaking_duration": 4.0,       # 2.0 s + 2.0 s = 4.0 s in in.shaking
        "settling_duration": 5.0,      # 5.0 s in in.shaking
        "dom_length": 0.3,
        "dom_width": 0.3,
        "dom_height": 1.2,
        "particle_properties": DEFAULT_TYPE_PROPERTIES.copy(),
    }

    raw_text = ""
    for fpath in unique_files:
        try:
            with open(fpath, "r", errors="ignore") as fh:
                raw_text += "\n" + fh.read()
        except Exception:
            pass

    if raw_text:
        # Particle diameter
        m_dia = re.findall(r"variable\s+particle_dia\s+equal\s+([\d\.\+eE\-]+)", raw_text)
        if m_dia:
            try:
                params["particle_dia"] = float(m_dia[-1])
                params["particle_radius"] = params["particle_dia"] / 2.0
            except ValueError:
                pass

        # Timestep
        m_dt = re.findall(r"(?:timestep\s+|variable\s+dt\s+equal\s+)([\d\.\+eE\-]+)", raw_text)
        if m_dt:
            try:
                params["dt"] = float(m_dt[-1])
            except ValueError:
                pass

        # Dumptime
        m_dump = re.findall(r"variable\s+dumptime\s+equal\s+([\d\.\+eE\-]+)", raw_text)
        if m_dump:
            try:
                params["dumptime"] = float(m_dump[-1])
            except ValueError:
                pass

        # Shaking amplitude
        m_amp = re.findall(r"variable\s+amp\s+equal\s+([\d\.\+eE\-]+|\$\{particle_dia\}/[\d\.]+)", raw_text)
        if m_amp:
            val_str = m_amp[-1]
            if "particle_dia" in val_str:
                div = float(val_str.split("/")[-1])
                params["shaking_amp"] = params["particle_dia"] / div
            else:
                try:
                    params["shaking_amp"] = float(val_str)
                except ValueError:
                    pass

        # Shaking frequency
        m_freq = re.findall(r"variable\s+freq\s+equal\s+([\d\.\+eE\-]+)", raw_text)
        if m_freq:
            try:
                params["shaking_freq"] = float(m_freq[-1])
            except ValueError:
                pass

    # Parse per-shaking cycle durations from ordered shaking scripts
    shaking_scripts = _discover_shaking_scripts(sim_dir)
    shaking_cycles = []
    for idx, spath in enumerate(shaking_scripts):
        try:
            stext = spath.read_text(errors="ignore")
        except Exception:
            continue
        var_secs = _extract_step_var_seconds(stext)
        shake_dur = params["shaking_duration"]
        settle_dur = params["settling_duration"]
        if "shear_wiggle" in stext:
            shake_block = stext[stext.find("shear_wiggle"):]
            unfix_idx = shake_block.find("unfix")
            if unfix_idx != -1:
                active_block = shake_block[:unfix_idx]
                post_block = shake_block[unfix_idx:]
                parsed_shake = _parse_run_steps_seconds(active_block, params["dt"], var_secs)
                parsed_settle = _parse_run_steps_seconds(post_block, params["dt"], var_secs)
                if parsed_shake > 0:
                    shake_dur = parsed_shake
                if parsed_settle > 0:
                    settle_dur = parsed_settle
        shaking_cycles.append({
            "cycle": idx + 1,
            "script": spath.name,
            "shaking_duration": shake_dur,
            "settling_duration": settle_dur,
            "shaking_amp": params["shaking_amp"],
            "shaking_freq": params["shaking_freq"],
        })

    if not shaking_cycles:
        # Fallback to single-cycle parsing from raw_text
        if raw_text and "shear_wiggle" in raw_text:
            shake_block = raw_text[raw_text.find("shear_wiggle"):]
            unfix_idx = shake_block.find("unfix")
            if unfix_idx != -1:
                parsed_shake = _parse_run_steps_seconds(shake_block[:unfix_idx], params["dt"])
                if parsed_shake > 0:
                    params["shaking_duration"] = parsed_shake
        shaking_cycles.append({
            "cycle": 1,
            "script": "default",
            "shaking_duration": params["shaking_duration"],
            "settling_duration": params["settling_duration"],
            "shaking_amp": params["shaking_amp"],
            "shaking_freq": params["shaking_freq"],
        })

    params["shaking_cycles"] = shaking_cycles
    params["shaking_duration"] = shaking_cycles[0]["shaking_duration"]
    params["settling_duration"] = shaking_cycles[0]["settling_duration"]

    # Try particle properties extraction
    if extract_particle_properties is not None:
        try:
            props, _ = extract_particle_properties(sim_dir)
            if props:
                for t, info in props.items():
                    if t not in params["particle_properties"]:
                        params["particle_properties"][t] = {}
                    params["particle_properties"][t].update(info)
        except Exception:
            pass

    return params


# ---------------------------------------------------------------------------
# Shaking Start Timestep & Dump-File Discovery
# ---------------------------------------------------------------------------
def _parse_log_timestep_range(log_path: Path) -> Optional[Tuple[int, int]]:
    """Extract (first_timestep, last_timestep) from a single LAMMPS/LIGGGHTS log file."""
    try:
        with open(log_path, "r", errors="ignore") as fh:
            content = fh.read()

        step_header_match = re.search(
            r"^\s*Step\s+Atoms.*?\n\s*(\d+)", content, re.MULTILINE
        )
        all_steps = re.findall(
            r"^\s*(\d+)\s+\d+\s+[\d\.\+eE\-]+", content, re.MULTILINE
        )
        if step_header_match:
            first_ts = int(step_header_match.group(1))
            last_ts = int(all_steps[-1]) if all_steps else first_ts
            return first_ts, last_ts
        elif all_steps:
            return int(all_steps[0]), int(all_steps[-1])
    except Exception:
        pass
    return None


def find_shaking_timestep_ranges(
    sim_dir: Union[str, Path],
    output_log: Optional[Union[str, Path]] = None,
) -> List[dict]:
    """
    Find timestep ranges `(first_ts, last_ts, log_path)` for all shaking log
    files (e.g., `output_shaking.txt`, `output_shaking_1.txt`, `output_shaking_2.txt`),
    sorted chronologically by `first_ts`.
    """
    sim_dir = Path(sim_dir)
    if output_log is not None:
        p = Path(output_log)
        target = p if p.is_file() else (sim_dir / output_log)
        if target.is_file():
            rng = _parse_log_timestep_range(target)
            if rng:
                return [{"first_ts": rng[0], "last_ts": rng[1], "log_path": target}]
        return []

    search_dirs = [sim_dir, sim_dir.parent]
    shaking_candidates = []
    fallback_candidates = []
    for d in search_dirs:
        if not d.is_dir():
            continue
        for pat in ("output_shaking*.txt", "*shaking*.txt", "*shaking*.log"):
            shaking_candidates.extend(sorted(d.glob(pat)))
        for pat in ("output*.txt", "log.*"):
            for f in sorted(d.glob(pat)):
                name_low = f.name.lower()
                if "bed_prep" not in name_low and "preparation" not in name_low:
                    fallback_candidates.append(f)

    candidates = shaking_candidates if shaking_candidates else fallback_candidates
    unique_logs = []
    for f in candidates:
        if f.is_file() and not f.name.startswith("._") and f not in unique_logs:
            unique_logs.append(f)

    ranges = []
    for log_path in unique_logs:
        rng = _parse_log_timestep_range(log_path)
        if rng:
            ranges.append({"first_ts": rng[0], "last_ts": rng[1], "log_path": log_path})

    ranges.sort(key=lambda r: r["first_ts"])
    return ranges


def find_shaking_start_timestep(
    sim_dir: Union[str, Path],
    output_log: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[int], Optional[int], Optional[Path]]:
    """
    Find the initial and final timesteps across all shaking simulation logs
    (`output_shaking*.txt` or other simulation log files), so that earlier
    bed-preparation dumps are excluded while all sequential shaking runs are kept.

    Parameters
    ----------
    sim_dir : str or Path
        Simulation directory.
    output_log : str or Path, optional
        Specific path or filename for the shaking output log.

    Returns
    -------
    tuple: (first_timestep, last_timestep, log_path)
    """
    ranges = find_shaking_timestep_ranges(sim_dir, output_log=output_log)
    if not ranges:
        return None, None, None
    first_ts = min(r["first_ts"] for r in ranges)
    last_ts = max(r["last_ts"] for r in ranges)
    return first_ts, last_ts, ranges[0]["log_path"]


def discover_dump_files(
    sim_dir: Union[str, Path],
    start_timestep: Optional[int] = None,
    end_timestep: Optional[int] = None,
    filter_by_output: bool = True,
    output_log: Optional[Union[str, Path]] = None,
) -> Tuple[List[Tuple[int, Path]], Optional[int], Optional[Path]]:
    """
    Find all dump files (e.g. results/dump*.post or dump*) within sim_dir.
    Filters out dumps before `start_timestep` (or before the first dump recorded in
    output_shaking.txt), so that earlier bed preparation dumps are excluded.

    Parameters
    ----------
    sim_dir : Path or str
        Simulation directory.
    start_timestep : int, optional
        Explicit earliest timestep to include. If None and filter_by_output=True,
        auto-detected from output_shaking.txt.
    end_timestep : int, optional
        Explicit latest timestep to include.
    filter_by_output : bool
        Whether to automatically check output_shaking.txt for the start timestep.
    output_log : Path or str, optional
        Specific shaking output log path.

    Returns
    -------
    tuple: (sorted_filtered_dumps, detected_start_ts, log_path)
        sorted_filtered_dumps: list of (timestep, file_path)
    """
    sim_dir = Path(sim_dir)
    results_dir = sim_dir / "results"
    target_dirs = [results_dir, sim_dir]

    found_files = []
    for d in target_dirs:
        if d.is_dir():
            for pat in ("dump*.post", "dump*.dump", "dump*"):
                for p in d.glob(pat):
                    if p.is_file() and not p.name.startswith("._") and p not in found_files:
                        found_files.append(p)

    if not found_files:
        return [], None, None

    def extract_index(p: Path) -> int:
        digits = re.findall(r"\d+", p.name)
        if digits:
            return int(digits[-1])
        return 0

    pairs = [(extract_index(p), p) for p in found_files]
    # Sort chronologically
    pairs.sort(key=lambda x: x[0])

    detected_start = start_timestep
    detected_end = end_timestep
    log_path = None

    if detected_start is None and filter_by_output:
        auto_start, auto_end, log_path = find_shaking_start_timestep(sim_dir, output_log=output_log)
        if auto_start is not None:
            detected_start = auto_start
        if detected_end is None and auto_end is not None:
            detected_end = auto_end

    # Filter out bed preparation dumps prior to shaking start
    if detected_start is not None:
        pairs = [p for p in pairs if p[0] >= detected_start]

    if detected_end is not None:
        pairs = [p for p in pairs if p[0] <= detected_end]

    return pairs, detected_start, log_path


# ---------------------------------------------------------------------------
# Top-Layer Extraction (2 Particle Diameters)
# ---------------------------------------------------------------------------
def extract_top_layer(
    df: pd.DataFrame,
    particle_diameter: Optional[float] = None,
    num_diameters: float = 2.0,
    method: str = "grid",
    grid_size: Optional[Tuple[int, int]] = None,
    exclude_types: Tuple[int, ...] = (1, 5),
) -> Tuple[pd.DataFrame, dict]:
    """
    Extract the top layer of the granular bed, defined as `num_diameters` * d
    (default: 2 particle diameters) below the bed surface.

    Parameters
    ----------
    df : pd.DataFrame
        Particle data containing columns 'x', 'y', 'z', 'type', and optionally 'radius'.
    particle_diameter : float, optional
        Particle diameter in meters. If None, inferred from df['radius'] or defaults to 0.01 m.
    num_diameters : float
        Thickness of top layer in units of particle diameter (default: 2.0).
    method : str
        'grid' (recommended for 3D beds): local 2D surface elevation per grid cell.
        'global_percentile': top layer measured from global 99th percentile of z.
        'global_max': top layer measured from global max(z).
    grid_size : tuple of int (nx, ny), optional
        Grid resolution for method='grid'. If None, computed so cell size ~ 2.5 * d.
    exclude_types : tuple of int
        Particle types to exclude (e.g. 1 for floor, 5 for wall). Default: (1, 5).

    Returns
    -------
    df_top : pd.DataFrame
        Subset of particles located in the top layer.
    surface_stats : dict
        Surface metrics: mean surface height, RMS roughness, layer thickness, bed bounds.
    """
    # 1. Filter out floor and wall particles
    if exclude_types:
        bed_df = df[~df["type"].isin(exclude_types)].copy()
    else:
        bed_df = df.copy()

    if len(bed_df) == 0:
        return bed_df, {"mean_surface_z": np.nan, "surface_roughness": np.nan, "layer_thickness": 0.0}

    # 2. Determine particle diameter
    if particle_diameter is None or particle_diameter <= 0:
        if "radius" in bed_df.columns:
            particle_diameter = float(bed_df["radius"].median() * 2.0)
        else:
            particle_diameter = 0.01  # Default 1 cm

    layer_thickness = num_diameters * particle_diameter
    r_offset = bed_df["radius"] if "radius" in bed_df.columns else (particle_diameter / 2.0)

    # 3. Surface calculation & top layer identification
    if method == "grid":
        x_min, x_max = bed_df["x"].min(), bed_df["x"].max()
        y_min, y_max = bed_df["y"].min(), bed_df["y"].max()

        x_span = max(x_max - x_min, particle_diameter)
        y_span = max(y_max - y_min, particle_diameter)

        if grid_size is None:
            # Cell size approximately 2.5 * particle diameter
            cell_size = 2.5 * particle_diameter
            nx = max(int(np.round(x_span / cell_size)), 4)
            ny = max(int(np.round(y_span / cell_size)), 4)
        else:
            nx, ny = grid_size

        x_bins = np.linspace(x_min - 1e-6, x_max + 1e-6, nx + 1)
        y_bins = np.linspace(y_min - 1e-6, y_max + 1e-6, ny + 1)

        bed_df["x_cell"] = np.clip(np.digitize(bed_df["x"], x_bins) - 1, 0, nx - 1)
        bed_df["y_cell"] = np.clip(np.digitize(bed_df["y"], y_bins) - 1, 0, ny - 1)

        # Particle top elevation: z + radius
        bed_df["z_top_coord"] = bed_df["z"] + r_offset

        # Local surface height per cell:
        # Use 98th percentile or max to prevent a rare splashed/airborne particle from distorting the cell
        cell_surface = bed_df.groupby(["x_cell", "y_cell"])["z_top_coord"].agg(
            lambda s: np.percentile(s, 98.0) if len(s) >= 10 else s.max()
        ).rename("local_surface_z")

        bed_df = bed_df.merge(cell_surface, on=["x_cell", "y_cell"], how="left")

        # Top layer condition: z >= local_surface_z - layer_thickness
        is_top = bed_df["z"] >= (bed_df["local_surface_z"] - layer_thickness)
        df_top = bed_df[is_top].copy()

        surface_vals = cell_surface.values
        mean_surface_z = float(np.mean(surface_vals)) if len(surface_vals) > 0 else float(bed_df["z"].max())
        surface_roughness = float(np.std(surface_vals)) if len(surface_vals) > 0 else 0.0

    elif method == "global_percentile":
        z_coords = bed_df["z"] + r_offset
        surface_z = float(np.percentile(z_coords, 99.0))
        is_top = bed_df["z"] >= (surface_z - layer_thickness)
        df_top = bed_df[is_top].copy()
        mean_surface_z = surface_z
        surface_roughness = float(np.std(df_top["z"])) if len(df_top) > 0 else 0.0

    else:  # global_max
        surface_z = float((bed_df["z"] + r_offset).max())
        is_top = bed_df["z"] >= (surface_z - layer_thickness)
        df_top = bed_df[is_top].copy()
        mean_surface_z = surface_z
        surface_roughness = float(np.std(df_top["z"])) if len(df_top) > 0 else 0.0

    surface_stats = {
        "particle_diameter": particle_diameter,
        "layer_thickness": layer_thickness,
        "mean_surface_z": mean_surface_z,
        "surface_roughness": surface_roughness,
        "total_bed_particles": len(bed_df),
        "top_layer_particles": len(df_top),
        "method": method,
    }

    return df_top, surface_stats


# ---------------------------------------------------------------------------
# Single-Frame Analysis
# ---------------------------------------------------------------------------
def analyze_frame_top_layer(
    df: pd.DataFrame,
    metadata: dict,
    sim_params: dict,
    num_diameters: float = 2.0,
    method: str = "grid",
    exclude_types: Tuple[int, ...] = (1, 5),
) -> dict:
    """
    Analyze the top layer of a single simulation dump frame.
    Calculates particle type compositions (counts, %, mass %), velocities,
    and surface characteristics.
    """
    particle_dia = sim_params.get("particle_dia", 0.01)
    df_top, surf_stats = extract_top_layer(
        df=df,
        particle_diameter=particle_dia,
        num_diameters=num_diameters,
        method=method,
        exclude_types=exclude_types,
    )

    bed_df = df[~df["type"].isin(exclude_types)] if exclude_types else df
    timestep = metadata.get("timestep", None)

    active_types = sorted(bed_df["type"].unique())
    has_mass = "mass" in df.columns
    has_vel = all(c in df.columns for c in ("vx", "vy", "vz"))

    n_total_top = len(df_top)
    n_total_bed = len(bed_df)

    mass_total_top = float(df_top["mass"].sum()) if (has_mass and n_total_top > 0) else np.nan
    mass_total_bed = float(bed_df["mass"].sum()) if (has_mass and n_total_bed > 0) else np.nan

    type_counts_top = df_top["type"].value_counts().to_dict()
    type_counts_bed = bed_df["type"].value_counts().to_dict()

    row = {
        "timestep": timestep,
        "total_bed_particles": n_total_bed,
        "top_particles": n_total_top,
        "top_fraction_of_bed": (n_total_top / n_total_bed) if n_total_bed > 0 else 0.0,
        "mean_surface_z": surf_stats["mean_surface_z"],
        "surface_roughness": surf_stats["surface_roughness"],
        "layer_thickness": surf_stats["layer_thickness"],
    }

    # Type-specific metrics
    for p_type in active_types:
        c_top = type_counts_top.get(p_type, 0)
        c_bed = type_counts_bed.get(p_type, 0)

        pct_top = (c_top / n_total_top * 100.0) if n_total_top > 0 else 0.0
        pct_bed = (c_bed / n_total_bed * 100.0) if n_total_bed > 0 else 0.0
        enrichment = (pct_top / pct_bed) if pct_bed > 0 else np.nan

        row[f"type_{p_type}_count"] = c_top
        row[f"type_{p_type}_pct"] = pct_top
        row[f"type_{p_type}_bed_pct"] = pct_bed
        row[f"type_{p_type}_enrichment"] = enrichment

        if has_mass:
            subset_top = df_top[df_top["type"] == p_type]
            m_top = float(subset_top["mass"].sum())
            m_pct_top = (m_top / mass_total_top * 100.0) if mass_total_top > 0 else 0.0
            row[f"type_{p_type}_mass"] = m_top
            row[f"type_{p_type}_mass_pct"] = m_pct_top

    # Kinematics in top layer
    if has_vel and n_total_top > 0:
        vx = df_top["vx"].values
        vy = df_top["vy"].values
        vz = df_top["vz"].values
        speeds = np.sqrt(vx**2 + vy**2 + vz**2)

        mean_vx, mean_vy, mean_vz = np.mean(vx), np.mean(vy), np.mean(vz)
        granular_temp = (np.var(vx) + np.var(vy) + np.var(vz)) / 3.0

        row["mean_vx"] = float(mean_vx)
        row["mean_vy"] = float(mean_vy)
        row["mean_vz"] = float(mean_vz)
        row["mean_speed"] = float(np.mean(speeds))
        row["granular_temperature"] = float(granular_temp)
    else:
        row["mean_vx"] = 0.0
        row["mean_vy"] = 0.0
        row["mean_vz"] = 0.0
        row["mean_speed"] = 0.0
        row["granular_temperature"] = 0.0

    return {
        "frame_metrics": row,
        "top_df": df_top,
        "active_types": active_types,
    }


# ---------------------------------------------------------------------------
# Phase Classification
# ---------------------------------------------------------------------------
def classify_phases(
    time_series_df: pd.DataFrame,
    shaking_start_time: Optional[float] = None,
    shaking_end_time: Optional[float] = None,
    shaking_windows: Optional[List[Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """
    Categorize each frame into shaking phases.
    For a single shaking window: 'before', 'during', 'after'.
    For multiple shaking windows: 'before', 'during_1', 'after_1', 'during_2', 'after_2', ...
    Also adds a 'shaking_cycle' integer column (0 for 'before', k for cycle k).
    """
    df = time_series_df.copy()
    if shaking_windows is None:
        if shaking_start_time is None or shaking_end_time is None:
            raise ValueError("Provide either shaking_windows or both shaking_start_time and shaking_end_time.")
        shaking_windows = [(float(shaking_start_time), float(shaking_end_time))]

    n_win = len(shaking_windows)
    t = df["time_sec"].values
    phases = np.empty(len(df), dtype=object)
    cycles = np.zeros(len(df), dtype=int)

    if n_win == 1:
        ws, we = shaking_windows[0]
        for idx, ti in enumerate(t):
            if ti <= ws + 1e-9:
                phases[idx] = "before"
                cycles[idx] = 0
            elif ti <= we + 1e-9:
                phases[idx] = "during"
                cycles[idx] = 1
            else:
                phases[idx] = "after"
                cycles[idx] = 1
    else:
        for idx, ti in enumerate(t):
            if ti <= shaking_windows[0][0] + 1e-9:
                phases[idx] = "before"
                cycles[idx] = 0
                continue
            assigned = False
            for k, (ws, we) in enumerate(shaking_windows, start=1):
                next_ws = shaking_windows[k][0] if k < n_win else float("inf")
                if ti <= we + 1e-9:
                    phases[idx] = f"during_{k}"
                    cycles[idx] = k
                    assigned = True
                    break
                elif ti <= next_ws + 1e-9:
                    phases[idx] = f"after_{k}"
                    cycles[idx] = k
                    assigned = True
                    break
            if not assigned:
                phases[idx] = f"after_{n_win}"
                cycles[idx] = n_win

    df["phase"] = phases
    df["shaking_cycle"] = cycles
    return df


def _resolve_shaking_windows(
    sim_params: dict,
    log_ranges: List[dict],
    first_ts: int,
    shaking_start: Optional[float] = None,
    shaking_end: Optional[float] = None,
    shaking_windows: Optional[List[Tuple[float, float]]] = None,
) -> List[Tuple[float, float]]:
    """
    Determine list of (t_start, t_end) shaking windows in seconds relative to first_ts.
    """
    if shaking_windows is not None and len(shaking_windows) > 0:
        return [(float(s), float(e)) for s, e in shaking_windows]

    if shaking_start is not None or shaking_end is not None:
        t_s = 0.0 if shaking_start is None else float(shaking_start)
        t_e = sim_params["shaking_duration"] if shaking_end is None else float(shaking_end)
        return [(t_s, t_e)]

    cycles = sim_params.get("shaking_cycles", [])
    dt = sim_params["dt"]
    if not cycles:
        return [(0.0, sim_params["shaking_duration"])]

    windows = []
    if len(log_ranges) == len(cycles) and len(log_ranges) > 0:
        for log_r, cyc in zip(log_ranges, cycles):
            t_s = max(0.0, (log_r["first_ts"] - first_ts) * dt)
            t_e = t_s + cyc["shaking_duration"]
            windows.append((t_s, t_e))
    else:
        t_cursor = 0.0
        for cyc in cycles:
            t_s = t_cursor
            t_e = t_s + cyc["shaking_duration"]
            windows.append((t_s, t_e))
            t_cursor = t_e + cyc["settling_duration"]
    return windows


def _ordered_phase_keys(n_shakings: int) -> List[Tuple[str, str]]:
    """Return ordered list of (internal_phase_key, display_label)."""
    if n_shakings <= 1:
        return [("before", "Before"), ("during", "During"), ("after", "After")]
    keys = [("before", "Before")]
    for k in range(1, n_shakings + 1):
        keys.append((f"during_{k}", f"During {k}"))
        keys.append((f"after_{k}", f"After {k}"))
    return keys


# ---------------------------------------------------------------------------
# Full Time-Series Analysis across Simulation Dumps
# ---------------------------------------------------------------------------
def analyze_bed_shaking_series(
    sim_dir: Union[str, Path],
    num_diameters: float = 2.0,
    method: str = "grid",
    shaking_start: Optional[float] = None,
    shaking_end: Optional[float] = None,
    shaking_windows: Optional[List[Tuple[float, float]]] = None,
    start_dump: Optional[int] = None,
    end_dump: Optional[int] = None,
    include_bed_prep: bool = False,
    output_log: Optional[Union[str, Path]] = None,
    step: int = 1,
    exclude_types: Tuple[int, ...] = (1, 5),
    verbose: bool = True,
) -> dict:
    """
    Analyze the entire dump sequence of a simulation in `bed_shaking/`.
    Extracts the top layer (2 particle diameters) for each dump, aggregates
    across phases (before, during, after shaking), and produces summary tables.

    By default, starts from the first dump recorded in `output_shaking.txt`
    (e.g. dump 2525000), ignoring earlier bed preparation dumps.

    Parameters
    ----------
    sim_dir : Path or str
        Path to simulation folder (e.g. `bed_shaking/seed_1`).
    num_diameters : float
        Top layer depth in particle diameters (default: 2.0).
    method : str
        'grid' (default), 'global_percentile', or 'global_max'.
    shaking_start : float, optional
        Start time of shaking in seconds (relative to first frame). Default: 0.0 s.
    shaking_end : float, optional
        End time of shaking in seconds (relative to first frame).
        If None, automatically read from in.shaking (typically 4.0 s).
    start_dump : int, optional
        Explicit earliest dump timestep to process. If None, auto-detected from output_shaking.txt.
    end_dump : int, optional
        Explicit latest dump timestep to process.
    include_bed_prep : bool
        If True, includes all bed preparation dumps prior to shaking. Default: False.
    output_log : str or Path, optional
        Path or filename of the shaking log file (default: searches for output_shaking.txt).
    step : int
        Process every Nth dump file (default: 1 = all files).
    exclude_types : tuple of int
        Exclude floor (1) and wall (5) particles.
    verbose : bool
        Print progress and summary tables.

    Returns
    -------
    results : dict
        Contains 'time_series_df', 'phase_summary_df', 'key_snapshots', 'sim_params'.
    """
    sim_dir = Path(sim_dir)
    sim_params = parse_shaking_simulation_parameters(sim_dir)
    dump_files, detected_start, log_found = discover_dump_files(
        sim_dir=sim_dir,
        start_timestep=start_dump,
        end_timestep=end_dump,
        filter_by_output=(not include_bed_prep),
        output_log=output_log,
    )

    if not dump_files:
        raise FileNotFoundError(
            f"No dump files found in '{sim_dir}' or '{sim_dir / 'results'}'. "
            "Please check that simulation results exist or run the simulation first."
        )

    # Subsample if step > 1
    selected_dumps = dump_files[::max(1, step)]
    total_frames = len(selected_dumps)

    if verbose:
        print("=" * 75)
        print(" BED SHAKING TOP-LAYER ANALYSIS")
        print("=" * 75)
        print(f"Simulation Directory : {sim_dir}")
        if log_found:
            print(f"Shaking Output Log   : {log_found.name}")
        if detected_start is not None and not include_bed_prep:
            print(f"Initial Shaking Dump : dump{detected_start}.post (bed-prep dumps excluded)")
        print(f"Processing Dumps     : {len(dump_files)} files (sampled {total_frames} with step={step})")
        print(f"Particle Diameter    : {sim_params['particle_dia']*1000:.1f} mm")
        print(f"Top Layer Thickness  : {num_diameters:.1f} d = {num_diameters * sim_params['particle_dia']*1000:.1f} mm")
        print(f"Shaking Frequency    : {sim_params['shaking_freq']} Hz | Amplitude: {sim_params['shaking_amp']*1000:.2f} mm")
        print(f"Extraction Method    : {method}")
        print("-" * 75)

    # Parse first timestep to set reference time t0
    first_ts = selected_dumps[0][0]
    dt = sim_params["dt"]

    log_ranges = find_shaking_timestep_ranges(sim_dir, output_log=output_log)
    resolved_windows = _resolve_shaking_windows(
        sim_params=sim_params,
        log_ranges=log_ranges,
        first_ts=first_ts,
        shaking_start=shaking_start,
        shaking_end=shaking_end,
        shaking_windows=shaking_windows,
    )

    records = []
    key_snapshots = {}  # Store sample particle DataFrames per phase

    for idx, (frame_num, dump_path) in enumerate(selected_dumps):
        try:
            meta, df = safe_read_dump(dump_path)
            ts = meta.get("timestep", frame_num)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed reading {dump_path.name}: {e}")
            continue

        time_sec = (ts - first_ts) * dt

        frame_res = analyze_frame_top_layer(
            df=df,
            metadata=meta,
            sim_params=sim_params,
            num_diameters=num_diameters,
            method=method,
            exclude_types=exclude_types,
        )

        row = frame_res["frame_metrics"]
        row["frame_index"] = idx
        row["file_path"] = str(dump_path)
        row["time_sec"] = time_sec
        records.append(row)

        # Retain representative snapshots for plotting 2D maps
        n_res = len(resolved_windows)
        if "before" not in key_snapshots and time_sec <= resolved_windows[0][0] + 1e-6:
            key_snapshots["before"] = {
                "time_sec": time_sec,
                "timestep": ts,
                "top_df": frame_res["top_df"],
                "full_df": df,
            }
        else:
            for k, (ws, we) in enumerate(resolved_windows, start=1):
                next_ws = resolved_windows[k][0] if k < n_res else float("inf")
                d_key = "during" if n_res == 1 else f"during_{k}"
                a_key = "after" if n_res == 1 else f"after_{k}"
                if ws < time_sec <= we + 1e-6:
                    if d_key not in key_snapshots and time_sec >= (ws + we) / 2.0:
                        key_snapshots[d_key] = {
                            "time_sec": time_sec,
                            "timestep": ts,
                            "top_df": frame_res["top_df"],
                            "full_df": df,
                        }
                    break
                elif we < time_sec <= next_ws + 1e-6:
                    # Continuously update after_k with the latest settled frame in that window
                    key_snapshots[a_key] = {
                        "time_sec": time_sec,
                        "timestep": ts,
                        "top_df": frame_res["top_df"],
                        "full_df": df,
                    }
                    break

        if verbose and (idx % max(1, total_frames // 10) == 0 or idx == total_frames - 1):
            pct_str = ""
            for p_type in (2, 3, 4):
                if f"type_{p_type}_pct" in row:
                    pct_str += f"T{p_type}:{row[f'type_{p_type}_pct']:.1f}% "
            print(f"  [{idx+1:3d}/{total_frames}] t={time_sec:6.3f}s | Top N={row['top_particles']:4d} | {pct_str}")

    df_series = pd.DataFrame(records)
    if df_series.empty:
        raise RuntimeError("No valid frames could be analyzed.")

    # Keep only shaking windows that actually occur within the available dump span
    t_max_data = float(df_series["time_sec"].max())
    active_windows = [w for w in resolved_windows if w[0] < t_max_data - 1e-6]
    if not active_windows:
        active_windows = [resolved_windows[0]]

    if len(resolved_windows) > 1 and len(active_windows) == 1:
        # Normalize *_1 snapshot keys when only the first shaking is present in dumps
        if "during_1" in key_snapshots and "during" not in key_snapshots:
            key_snapshots["during"] = key_snapshots.pop("during_1")
        if "after_1" in key_snapshots and "after" not in key_snapshots:
            key_snapshots["after"] = key_snapshots.pop("after_1")

    n_shakings = len(active_windows)
    t_start_shake = active_windows[0][0]
    t_end_shake = active_windows[-1][1]

    # Classify phases
    df_series = classify_phases(df_series, shaking_windows=active_windows)

    # Fill any missing key snapshots from df_series
    ordered_phases = _ordered_phase_keys(n_shakings)
    for p_key, _ in ordered_phases:
        if p_key not in key_snapshots:
            sub = df_series[df_series["phase"] == p_key]
            if not sub.empty:
                sel = sub.iloc[0] if p_key == "before" else (
                    sub.iloc[len(sub) // 2] if p_key.startswith("during") else sub.iloc[-1]
                )
                key_snapshots[p_key] = {"time_sec": sel["time_sec"], "timestep": sel["timestep"]}

    # -----------------------------------------------------------------------
    # Phase Aggregation Summary Table
    # -----------------------------------------------------------------------
    phase_summary_rows = []
    pct_cols = [c for c in df_series.columns if c.endswith("_pct") and not c.endswith("_mass_pct") and not c.endswith("_bed_pct")]
    types = [int(re.search(r"type_(\d+)_", c).group(1)) for c in pct_cols]

    for p_key, p_label in ordered_phases:
        phase_sub = df_series[df_series["phase"] == p_key]
        if phase_sub.empty:
            continue

        row_data = {
            "phase_key": p_key,
            "phase": p_label,
            "num_frames": len(phase_sub),
            "time_range_s": f"[{phase_sub['time_sec'].min():.2f}, {phase_sub['time_sec'].max():.2f}]",
            "mean_top_particles": phase_sub["top_particles"].mean(),
            "mean_surface_z_m": phase_sub["mean_surface_z"].mean(),
            "mean_roughness_m": phase_sub["surface_roughness"].mean(),
            "mean_speed_m_s": phase_sub["mean_speed"].mean(),
            "mean_granular_temp": phase_sub["granular_temperature"].mean(),
        }

        # For "before" and "after*" phases, record settled snapshot (first for before, last for after)
        snapshot_row = phase_sub.iloc[0] if p_key == "before" else phase_sub.iloc[-1]

        for p_type in types:
            col_p = f"type_{p_type}_pct"
            col_m = f"type_{p_type}_mass_pct"
            col_e = f"type_{p_type}_enrichment"

            row_data[f"type_{p_type}_pct_mean"] = phase_sub[col_p].mean()
            row_data[f"type_{p_type}_pct_std"] = phase_sub[col_p].std()
            row_data[f"type_{p_type}_pct_snapshot"] = snapshot_row[col_p]

            if col_m in phase_sub.columns:
                row_data[f"type_{p_type}_mass_pct_mean"] = phase_sub[col_m].mean()
                row_data[f"type_{p_type}_mass_pct_snapshot"] = snapshot_row[col_m]

            if col_e in phase_sub.columns:
                row_data[f"type_{p_type}_enrichment_mean"] = phase_sub[col_e].mean()

        phase_summary_rows.append(row_data)

    df_phase_summary = pd.DataFrame(phase_summary_rows)

    # Compute Cumulative Net Segregation Change: (Final After - Before)
    segregation_delta = {}
    last_after_key = "after" if n_shakings == 1 else f"after_{n_shakings}"
    if "before" in df_phase_summary["phase_key"].values and last_after_key in df_phase_summary["phase_key"].values:
        before_row = df_phase_summary[df_phase_summary["phase_key"] == "before"].iloc[0]
        after_row = df_phase_summary[df_phase_summary["phase_key"] == last_after_key].iloc[0]

        for p_type in types:
            d_pct = after_row[f"type_{p_type}_pct_snapshot"] - before_row[f"type_{p_type}_pct_snapshot"]
            segregation_delta[f"type_{p_type}_delta_pct"] = d_pct
            if f"type_{p_type}_mass_pct_snapshot" in after_row and f"type_{p_type}_mass_pct_snapshot" in before_row:
                d_mass = after_row[f"type_{p_type}_mass_pct_snapshot"] - before_row[f"type_{p_type}_mass_pct_snapshot"]
                segregation_delta[f"type_{p_type}_delta_mass_pct"] = d_mass

    # Compute Per-Shaking Comparison Table (comparing each shaking cycle)
    props = sim_params.get("particle_properties", DEFAULT_TYPE_PROPERTIES)
    shaking_comp_rows = []
    if "before" in df_phase_summary["phase_key"].values:
        init_row = df_phase_summary[df_phase_summary["phase_key"] == "before"].iloc[0]
        for k in range(1, n_shakings + 1):
            pre_key = "before" if k == 1 else f"after_{k - 1}"
            post_key = "after" if n_shakings == 1 else f"after_{k}"
            if pre_key not in df_phase_summary["phase_key"].values or post_key not in df_phase_summary["phase_key"].values:
                continue
            pre_row = df_phase_summary[df_phase_summary["phase_key"] == pre_key].iloc[0]
            post_row = df_phase_summary[df_phase_summary["phase_key"] == post_key].iloc[0]
            ws, we = active_windows[k - 1]

            for p_type in types:
                name = props.get(p_type, {}).get("name", f"Type {p_type}")
                pre_pct = pre_row[f"type_{p_type}_pct_snapshot"]
                post_pct = post_row[f"type_{p_type}_pct_snapshot"]
                init_pct = init_row[f"type_{p_type}_pct_snapshot"]

                comp_entry = {
                    "shaking_cycle": k,
                    "shake_window_s": f"[{ws:.2f}, {we:.2f}]",
                    "pre_phase": pre_row["phase"],
                    "post_phase": post_row["phase"],
                    "type": p_type,
                    "name": name,
                    "before_pct": pre_pct,
                    "after_pct": post_pct,
                    "delta_pct": post_pct - pre_pct,
                    "cumulative_delta_pct": post_pct - init_pct,
                    "surface_z_delta_mm": (post_row["mean_surface_z_m"] - pre_row["mean_surface_z_m"]) * 1000.0,
                    "roughness_delta_mm": (post_row["mean_roughness_m"] - pre_row["mean_roughness_m"]) * 1000.0,
                }
                if f"type_{p_type}_mass_pct_snapshot" in post_row and f"type_{p_type}_mass_pct_snapshot" in pre_row:
                    pre_m = pre_row[f"type_{p_type}_mass_pct_snapshot"]
                    post_m = post_row[f"type_{p_type}_mass_pct_snapshot"]
                    init_m = init_row[f"type_{p_type}_mass_pct_snapshot"]
                    comp_entry["before_mass_pct"] = pre_m
                    comp_entry["after_mass_pct"] = post_m
                    comp_entry["delta_mass_pct"] = post_m - pre_m
                    comp_entry["cumulative_delta_mass_pct"] = post_m - init_m

                shaking_comp_rows.append(comp_entry)

    df_shaking_comparison = pd.DataFrame(shaking_comp_rows)

    results = {
        "time_series_df": df_series,
        "phase_summary_df": df_phase_summary,
        "shaking_comparison_df": df_shaking_comparison,
        "segregation_delta": segregation_delta,
        "key_snapshots": key_snapshots,
        "sim_params": sim_params,
        "num_diameters": num_diameters,
        "method": method,
        "types": types,
        "t_start_shake": t_start_shake,
        "t_end_shake": t_end_shake,
        "shaking_windows": active_windows,
        "n_shakings": n_shakings,
    }

    if verbose:
        print_analysis_summary(results)

    return results


# ---------------------------------------------------------------------------
# Formatted Terminal Summary Report
# ---------------------------------------------------------------------------
def print_analysis_summary(results: dict):
    """Print readable summary tables of the top layer analysis."""
    df_phase = results["phase_summary_df"]
    sim_params = results["sim_params"]
    props = sim_params.get("particle_properties", DEFAULT_TYPE_PROPERTIES)
    types = results["types"]

    print("\n" + "=" * 75)
    print("        TOP LAYER COMPOSITION SUMMARY: BEFORE vs DURING vs AFTER")
    print(f"        (Top Layer defined as {results['num_diameters']:.1f} Particle Diameters = {results['num_diameters']*sim_params['particle_dia']*1000:.1f} mm)")
    print("=" * 75)

    headers = ["Phase", "Time Span (s)", "Frames", "Mean Z (m)", "Roughness (mm)"]
    for t in types:
        name = props.get(t, {}).get("name", f"Type {t}")
        headers.append(f"{name} (%)")

    table_rows = []
    for _, r in df_phase.iterrows():
        row_str = [
            f"{r['phase']:<8}",
            f"{r['time_range_s']:<14}",
            f"{int(r['num_frames']):<6}",
            f"{r['mean_surface_z_m']:.4f}",
            f"{r['mean_roughness_m']*1000:.2f}",
        ]
        for t in types:
            pct_val = r.get(f"type_{t}_pct_snapshot", r.get(f"type_{t}_pct_mean", np.nan))
            row_str.append(f"{pct_val:6.2f}%")
        table_rows.append(row_str)

    # Print table header
    col_widths = [10, 16, 8, 12, 16] + [16] * len(types)
    header_line = "".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    for row in table_rows:
        print("".join(f"{v:<{w}}" for v, w in zip(row, col_widths)))
    print("-" * len(header_line))

    # Print Per-Shaking Comparison when multiple shakings exist
    df_comp = results.get("shaking_comparison_df", pd.DataFrame())
    n_shakings = results.get("n_shakings", 1)
    if n_shakings >= 2 and not df_comp.empty:
        print("\nSHAKING-TO-SHAKING COMPARISON (TOP LAYER SETTLED SNAPSHOTS):")
        comp_headers = ["Shake", "Window (s)", "Species", "Before (%)", "After (%)", "Δ Shake (%)", "Cumul. Δ (%)"]
        comp_widths = [8, 15, 12, 12, 12, 13, 13]
        print("".join(f"{h:<{w}}" for h, w in zip(comp_headers, comp_widths)))
        print("-" * sum(comp_widths))
        for _, cr in df_comp.iterrows():
            vals = [
                f"#{int(cr['shaking_cycle']):<7}",
                f"{cr['shake_window_s']:<15}",
                f"{cr['name']:<12}",
                f"{cr['before_pct']:6.2f}%     ",
                f"{cr['after_pct']:6.2f}%     ",
                f"{cr['delta_pct']:+6.2f}%      ",
                f"{cr['cumulative_delta_pct']:+6.2f}%      ",
            ]
            print("".join(vals))
        print("-" * sum(comp_widths))

    # Print Segregation Deltas
    deltas = results.get("segregation_delta", {})
    if deltas:
        label_delta = "CUMULATIVE NET SEGREGATION CHANGE (Final After - Initial Before)" if n_shakings >= 2 else "NET SEGREGATION CHANGE IN TOP LAYER (After Shaking - Before Shaking)"
        print(f"\n{label_delta}:")
        for t in types:
            name = props.get(t, {}).get("name", f"Type {t}")
            dens = props.get(t, {}).get("density", np.nan)
            d_pct = deltas.get(f"type_{t}_delta_pct", 0.0)
            direction = "▲ Enriched" if d_pct > 0.5 else ("▼ Depleted" if d_pct < -0.5 else "≈ Unchanged")
            print(f"  • {name:<10} (ρ = {dens:4.0f} kg/m³): {d_pct:+6.2f}%  [{direction}]")
    print("=" * 75 + "\n")


# ---------------------------------------------------------------------------
# Publication-Quality Plotting
# ---------------------------------------------------------------------------
def plot_top_layer_time_series(results: dict, save_path: Optional[Union[str, Path]] = None, show: bool = False):
    """
    Plot time series of:
    1. Top-layer percentage for each particle type over time, with shaded phase regions.
    2. Bed surface elevation dilation and granular temperature / kinetic activity.
    """
    df = results["time_series_df"]
    sim_params = results["sim_params"]
    props = sim_params.get("particle_properties", DEFAULT_TYPE_PROPERTIES)
    types = results["types"]
    shaking_windows = results.get("shaking_windows") or [(results["t_start_shake"], results["t_end_shake"])]
    n_shakings = len(shaking_windows)

    fig_w = 11 if n_shakings <= 1 else min(16, 10 + 2.5 * n_shakings)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_w, 8), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0]})

    # Shaded phase bands
    t_min = df["time_sec"].min()
    t_max = df["time_sec"].max()

    phase_bands = []
    if t_min < shaking_windows[0][0]:
        phase_bands.append((t_min, shaking_windows[0][0], "#f0f3f4", "Before Shaking"))

    for k, (ws, we) in enumerate(shaking_windows, start=1):
        shake_lbl = (
            f"Shaking ({sim_params['shaking_freq']} Hz, {sim_params['shaking_amp']*1000:.1f} mm)"
            if n_shakings == 1
            else f"Shake {k}"
        )
        phase_bands.append((ws, min(we, t_max), "#fef9e7", shake_lbl))
        next_ws = shaking_windows[k][0] if k < n_shakings else t_max
        settle_lbl = "Settling / After Shaking" if n_shakings == 1 else f"Settling {k}"
        if we < min(next_ws, t_max):
            phase_bands.append((we, min(next_ws, t_max), "#eafaf1", settle_lbl))

    for ax in (ax1, ax2):
        for start, end, color, label in phase_bands:
            if start < end:
                ax.axvspan(start, end, color=color, alpha=0.7, zorder=0)
        if n_shakings >= 2:
            for ws, we in shaking_windows:
                for bnd in (ws, we):
                    if t_min < bnd < t_max:
                        ax.axvline(bnd, color="#888888", linestyle="--", linewidth=0.9, alpha=0.7, zorder=1)

    # 1. Top Layer Composition vs Time
    for p_type in types:
        info = props.get(p_type, {})
        name = info.get("name", f"Type {p_type}")
        color = info.get("color", None)
        col_pct = f"type_{p_type}_pct"

        if col_pct in df.columns:
            ax1.plot(
                df["time_sec"],
                df[col_pct],
                label=f"{name} ({info.get('density', 0):.0f} kg/m³)",
                color=color,
                linewidth=2.2,
                alpha=0.9,
            )

    ax1.set_ylabel("Top Layer Percentage (%)", fontsize=12, fontweight="bold")
    ax1.set_ylim(-2, 102)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.9, title="Particle Species")
    ax1.set_title(
        f"Top Layer ({results['num_diameters']:.1f}d) Composition Evolution Before, During & After Shaking",
        fontsize=13,
        fontweight="bold",
    )

    # Add phase annotations above ax1
    y_top = 92
    for start, end, _, label in phase_bands:
        if start < end:
            mid = (start + end) / 2.0
            ax1.text(
                mid,
                y_top,
                label,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#555555",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc", alpha=0.8),
            )

    # 2. Bed Surface Elevation & Kinematics
    ax2_twin = ax2.twinx()

    line_surf = ax2.plot(
        df["time_sec"],
        df["mean_surface_z"] * 1000,
        color="#2c3e50",
        linewidth=2.0,
        label="Bed Surface Height (mm)",
    )
    line_temp = ax2_twin.plot(
        df["time_sec"],
        df["granular_temperature"] * 1e4,
        color="#e74c3c",
        linewidth=1.8,
        linestyle="-.",
        label="Granular Temp (x10⁻⁴ m²/s²)",
    )

    ax2.set_xlabel("Physical Time (s)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Mean Surface Height (mm)", color="#2c3e50", fontsize=11, fontweight="bold")
    ax2_twin.set_ylabel("Granular Temperature", color="#e74c3c", fontsize=11, fontweight="bold")

    lines = line_surf + line_temp
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="upper right", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_top_layer_phase_comparison(
    results: dict,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
):
    """
    Grouped bar chart comparing the top-layer composition (number percentage
    and mass percentage) across Before, During, and After shaking phases.
    """
    df_phase = results["phase_summary_df"]
    sim_params = results["sim_params"]
    props = sim_params.get("particle_properties", DEFAULT_TYPE_PROPERTIES)
    types = results["types"]

    phases = df_phase["phase"].tolist()
    n_phases = len(phases)
    n_types = len(types)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    bar_width = 0.8 / max(n_types, 1)
    x_positions = np.arange(n_phases)

    # 1. Number Percentage
    for i, p_type in enumerate(types):
        info = props.get(p_type, {})
        name = info.get("name", f"Type {p_type}")
        color = info.get("color", None)

        vals = []
        for _, r in df_phase.iterrows():
            v = r.get(f"type_{p_type}_pct_snapshot", r.get(f"type_{p_type}_pct_mean", 0.0))
            vals.append(v)

        offset = (i - n_types / 2.0 + 0.5) * bar_width
        bars = ax1.bar(
            x_positions + offset,
            vals,
            width=bar_width,
            label=f"{name} ({info.get('density', 0):.0f} kg/m³)",
            color=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )
        # Value label on top of bar
        for bar in bars:
            h = bar.get_height()
            if h > 3:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 1.0,
                    f"{h:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(phases, fontsize=11, fontweight="bold")
    ax1.set_ylabel("Number Percentage in Top Layer (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.set_title("Top Layer Composition (Number %)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Mass Percentage
    for i, p_type in enumerate(types):
        info = props.get(p_type, {})
        name = info.get("name", f"Type {p_type}")
        color = info.get("color", None)

        vals_m = []
        for _, r in df_phase.iterrows():
            v = r.get(f"type_{p_type}_mass_pct_snapshot", r.get(f"type_{p_type}_mass_pct_mean", 0.0))
            vals_m.append(v)

        offset = (i - n_types / 2.0 + 0.5) * bar_width
        bars_m = ax2.bar(
            x_positions + offset,
            vals_m,
            width=bar_width,
            label=f"{name}",
            color=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )
        for bar in bars_m:
            h = bar.get_height()
            if h > 3:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 1.0,
                    f"{h:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(phases, fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mass Percentage in Top Layer (%)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.set_title("Top Layer Composition (Mass %)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle(
        f"Top Layer Comparison Across Phases (Depth = 2 Particle Diameters)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_shaking_comparison(
    results: dict,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
):
    """
    Side-by-side comparison across multiple shaking cycles:
    1. Settled top-layer composition (Number %) at Initial, After Shake 1, After Shake 2, ...
    2. Per-shaking incremental change (Δ%) vs cumulative change (Δ%) for each species.
    """
    df_comp = results.get("shaking_comparison_df", pd.DataFrame())
    df_phase = results["phase_summary_df"]
    if df_comp.empty or results.get("n_shakings", 1) < 2:
        return

    sim_params = results["sim_params"]
    props = sim_params.get("particle_properties", DEFAULT_TYPE_PROPERTIES)
    types = results["types"]
    n_shakings = results["n_shakings"]

    # Settled states: Before, After 1, After 2, ...
    settled_keys = ["before"] + [f"after_{k}" for k in range(1, n_shakings + 1)]
    settled_rows = [
        df_phase[df_phase["phase_key"] == k].iloc[0]
        for k in settled_keys
        if k in df_phase["phase_key"].values
    ]
    if len(settled_rows) < 2:
        return

    fig_w = max(12.5, 9.5 + 1.5 * n_shakings)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, 5.5))

    # Panel 1: Settled Composition Across Shaking Cycles
    state_labels = ["Initial\n(Before)" if r["phase_key"] == "before" else f"After\nShake {r['phase_key'].split('_')[1]}" for r in settled_rows]
    x_pos = np.arange(len(settled_rows))
    n_types = len(types)
    bar_w = 0.78 / max(n_types, 1)

    for i, p_type in enumerate(types):
        info = props.get(p_type, {})
        name = info.get("name", f"Type {p_type}")
        color = info.get("color", None)
        vals = [r[f"type_{p_type}_pct_snapshot"] for r in settled_rows]
        offset = (i - n_types / 2.0 + 0.5) * bar_w
        bars = ax1.bar(
            x_pos + offset,
            vals,
            width=bar_w,
            label=f"{name} ({info.get('density', 0):.0f} kg/m³)",
            color=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )
        for bar in bars:
            h = bar.get_height()
            if h > 2.5:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 1.0,
                    f"{h:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(state_labels, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Top Layer Number Percentage (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 108)
    ax1.set_title("Settled Top-Layer Composition After Each Shaking", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 2: Incremental (per shake) vs Cumulative Delta (%)
    delta_groups = [f"Shake {k}\n(Δ)" for k in range(1, n_shakings + 1)] + ["Cumulative\n(Total Δ)"]
    x_d = np.arange(len(delta_groups))
    for i, p_type in enumerate(types):
        info = props.get(p_type, {})
        name = info.get("name", f"Type {p_type}")
        color = info.get("color", None)
        sub_c = df_comp[df_comp["type"] == p_type].sort_values("shaking_cycle")
        inc_vals = sub_c["delta_pct"].tolist()
        cum_val = sub_c["cumulative_delta_pct"].iloc[-1] if not sub_c.empty else 0.0
        vals_d = inc_vals + [cum_val]

        offset = (i - n_types / 2.0 + 0.5) * bar_w
        bars_d = ax2.bar(
            x_d + offset,
            vals_d,
            width=bar_w,
            label=name,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )
        for bar in bars_d:
            h = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            pad = 0.8 if h >= 0 else -0.8
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + pad,
                f"{h:+.1f}%",
                ha="center",
                va=va,
                fontsize=8,
                fontweight="bold",
            )

    ax2.axhline(0.0, color="black", linewidth=1.0)
    ax2.set_xticks(x_d)
    ax2.set_xticklabels(delta_groups, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Change in Top Layer (% pts)", fontsize=11, fontweight="bold")
    ax2.set_title("Incremental vs. Cumulative Segregation Change", fontsize=12, fontweight="bold")
    ax2.legend(loc="best", framealpha=0.9)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle(
        f"Multi-Shaking Top-Layer Comparison (Depth = {results['num_diameters']:.1f}d)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def plot_top_layer_spatial_distribution(
    results: dict,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
):
    """
    2D Top-down scatter map of particle types located in the top layer
    at key benchmark snapshots across all shaking cycles.
    """
    key_snaps = results.get("key_snapshots", {})
    n_shakings = results.get("n_shakings", 1)
    ordered_keys = [k for k, _ in _ordered_phase_keys(n_shakings)]
    label_map = dict(_ordered_phase_keys(n_shakings))
    available_phases = [p for p in ordered_keys if p in key_snaps and "top_df" in key_snaps[p]]

    if not available_phases:
        return

    sim_params = results["sim_params"]
    props = sim_params.get("particle_properties", DEFAULT_TYPE_PROPERTIES)

    n_cols = len(available_phases)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4.5), squeeze=False)
    axes = axes[0]

    for idx, phase_key in enumerate(available_phases):
        ax = axes[idx]
        snap_info = key_snaps[phase_key]
        top_df = snap_info["top_df"]
        t_sec = snap_info["time_sec"]

        # Scatter plot colored by particle type
        for p_type in sorted(top_df["type"].unique()):
            sub = top_df[top_df["type"] == p_type]
            info = props.get(p_type, {})
            name = info.get("name", f"Type {p_type}")
            color = info.get("color", None)

            ax.scatter(
                sub["x"] * 1000,
                sub["y"] * 1000,
                s=25,
                color=color,
                label=f"{name}",
                alpha=0.75,
                edgecolors="none",
            )

        phase_title = label_map.get(phase_key, phase_key.capitalize())
        ax.set_title(f"{phase_title} (t = {t_sec:.2f} s)\nTop Layer N = {len(top_df)}", fontsize=11, fontweight="bold")
        ax.set_xlabel("X Position (mm)", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Y Position (mm)", fontsize=10)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.4)

    plt.suptitle("Top Layer (2d) Spatial Particle Type Distribution", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Saving Results
# ---------------------------------------------------------------------------
def save_analysis_results(results: dict, output_dir: Union[str, Path]) -> dict:
    """Save all tables and publication plots to output directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save CSVs
    ts_csv = out_dir / "top_layer_time_series.csv"
    results["time_series_df"].to_csv(ts_csv, index=False)

    summary_csv = out_dir / "top_layer_phase_summary.csv"
    results["phase_summary_df"].to_csv(summary_csv, index=False)

    saved_figs = []

    # 2. Save Plots
    fig_ts = out_dir / "top_layer_composition_vs_time.png"
    plot_top_layer_time_series(results, save_path=fig_ts)
    saved_figs.append(fig_ts)

    fig_bar = out_dir / "top_layer_phase_comparison.png"
    plot_top_layer_phase_comparison(results, save_path=fig_bar)
    saved_figs.append(fig_bar)

    fig_map = out_dir / "top_layer_spatial_distribution.png"
    plot_top_layer_spatial_distribution(results, save_path=fig_map)
    saved_figs.append(fig_map)

    n_shakings = results.get("n_shakings", 1)
    df_comp = results.get("shaking_comparison_df", pd.DataFrame())
    comp_csv = None
    if n_shakings >= 2 and not df_comp.empty:
        comp_csv = out_dir / "top_layer_shaking_comparison.csv"
        df_comp.to_csv(comp_csv, index=False)
        fig_comp = out_dir / "top_layer_shaking_comparison.png"
        plot_shaking_comparison(results, save_path=fig_comp)
        saved_figs.append(fig_comp)

    # 3. Save Markdown / Text Report
    report_file = out_dir / "top_layer_analysis_report.md"
    with open(report_file, "w") as f:
        f.write("# Bed Shaking Top Layer Analysis Report\n\n")
        f.write(f"- **Top Layer Thickness**: {results['num_diameters']:.1f} particle diameters "
                f"({results['num_diameters']*results['sim_params']['particle_dia']*1000:.1f} mm)\n")
        f.write(f"- **Extraction Method**: {results['method']}\n")
        windows = results.get("shaking_windows") or [(results["t_start_shake"], results["t_end_shake"])]
        win_str = ", ".join(f"Shake {i}: {ws:.2f} s to {we:.2f} s" for i, (ws, we) in enumerate(windows, 1))
        f.write(f"- **Shaking Windows**: {win_str}\n\n")

        f.write("## Phase Summary Table\n\n")
        df_p = results["phase_summary_df"]
        try:
            f.write(df_p.to_markdown(index=False))
        except Exception:
            cols = list(df_p.columns)
            header_str = "| " + " | ".join(cols) + " |\n"
            sep_str = "| " + " | ".join(["---"] * len(cols)) + " |\n"
            f.write(header_str + sep_str)
            for _, r in df_p.iterrows():
                row_vals = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in r.values]
                f.write("| " + " | ".join(row_vals) + " |\n")
        f.write("\n\n")

        if n_shakings >= 2 and not df_comp.empty:
            f.write("## Shaking-to-Shaking Comparison\n\n")
            try:
                f.write(df_comp.to_markdown(index=False))
            except Exception:
                cols = list(df_comp.columns)
                f.write("| " + " | ".join(cols) + " |\n")
                f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
                for _, r in df_comp.iterrows():
                    row_vals = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in r.values]
                    f.write("| " + " | ".join(row_vals) + " |\n")
            f.write("\n\n")

        deltas = results.get("segregation_delta", {})
        if deltas:
            f.write("## Net Segregation (Final After - Initial Before)\n\n")
            for t in results["types"]:
                info = results["sim_params"].get("particle_properties", {}).get(t, {})
                name = info.get("name", f"Type {t}")
                d_pct = deltas.get(f"type_{t}_delta_pct", 0.0)
                f.write(f"- **{name}** (Type {t}): {d_pct:+6.2f}%\n")
            f.write("\n")

    print(f"\nAll analysis outputs successfully saved to: {out_dir}")
    out_dict = {
        "time_series_csv": ts_csv,
        "phase_summary_csv": summary_csv,
        "figures": saved_figs,
        "report": report_file,
    }
    if comp_csv is not None:
        out_dict["shaking_comparison_csv"] = comp_csv
    return out_dict


# ---------------------------------------------------------------------------
# Synthetic Demo / Self-Test Generator
# ---------------------------------------------------------------------------
def generate_synthetic_shaking_dataset(
    output_dir: Union[str, Path],
    num_particles: int = 12000,
    box_size: float = 0.3,
    particle_dia: float = 0.01,
    dt: float = 4e-6,
    num_frames: int = 25,
    n_shakings: int = 1,
) -> Path:
    """
    Generate synthetic dump files mimicking a single or multi-cycle bed shaking
    simulation to verify pipeline correctness.
    """
    out_dir = Path(output_dir) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    radius = particle_dia / 2.0
    densities = {2: 7870.0, 3: 4900.0, 4: 2500.0}

    n_per_type = num_particles // 3
    types = np.array([2] * n_per_type + [3] * n_per_type + [4] * n_per_type)
    np.random.shuffle(types)

    x = np.random.uniform(0.01, box_size - 0.01, num_particles)
    y = np.random.uniform(0.01, box_size - 0.01, num_particles)
    z = np.random.uniform(0.02, 0.35, num_particles)

    # Cycle 1: shake [0, 4.0], settle [4.0, 9.0]
    # Cycle 2 (if n_shakings >= 2): shake [9.0, 13.0], settle [13.0, 16.0]
    total_time = 9.0 if n_shakings <= 1 else 16.0
    time_points = np.linspace(0.0, total_time, num_frames)

    for frame_idx, t in enumerate(time_points):
        ts = int(t / dt)
        z_current = z.copy()
        rocky_mask = (types == 4)
        kamacite_mask = (types == 2)

        if 0.0 < t <= 4.0:
            progress = t / 4.0
            z_current[rocky_mask] += 0.04 * progress * np.random.uniform(0.7, 1.2, np.sum(rocky_mask))
            z_current[kamacite_mask] -= 0.03 * progress * np.random.uniform(0.7, 1.2, np.sum(kamacite_mask))
            z_current += 0.002 * np.sin(2 * np.pi * 5.0 * t) + 0.015 * progress
            vx = np.random.normal(0, 0.05, num_particles)
            vy = np.random.normal(0, 0.05, num_particles)
            vz = np.random.normal(0, 0.15, num_particles)
        elif 4.0 < t <= 9.0 or (n_shakings <= 1 and t > 4.0):
            z_current[rocky_mask] += 0.04
            z_current[kamacite_mask] -= 0.03
            vx = np.zeros(num_particles)
            vy = np.zeros(num_particles)
            vz = np.zeros(num_particles)
        elif n_shakings >= 2 and 9.0 < t <= 13.0:
            progress = (t - 9.0) / 4.0
            z_current[rocky_mask] += 0.04 + 0.025 * progress * np.random.uniform(0.7, 1.2, np.sum(rocky_mask))
            z_current[kamacite_mask] -= 0.03 + 0.02 * progress * np.random.uniform(0.7, 1.2, np.sum(kamacite_mask))
            z_current += 0.002 * np.sin(2 * np.pi * 5.0 * (t - 9.0)) + 0.012 * progress
            vx = np.random.normal(0, 0.05, num_particles)
            vy = np.random.normal(0, 0.05, num_particles)
            vz = np.random.normal(0, 0.15, num_particles)
        elif n_shakings >= 2 and t > 13.0:
            z_current[rocky_mask] += 0.065
            z_current[kamacite_mask] -= 0.05
            vx = np.zeros(num_particles)
            vy = np.zeros(num_particles)
            vz = np.zeros(num_particles)
        else:
            vx = np.zeros(num_particles)
            vy = np.zeros(num_particles)
            vz = np.zeros(num_particles)

        # Particle mass
        masses = [
            (4.0 / 3.0) * np.pi * (radius**3) * densities[int(t_i)]
            for t_i in types
        ]

        df_dump = pd.DataFrame({
            "id": np.arange(1, num_particles + 1),
            "type": types,
            "x": x,
            "y": y,
            "z": z_current,
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "radius": radius,
            "fx": 0.0,
            "fy": 0.0,
            "fz": 0.0,
            "mass": masses,
        })

        # Save in standard LIGGGHTS / LAMMPS dump format
        dump_file = out_dir / f"dump{ts:08d}.post"
        with open(dump_file, "w") as f:
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{ts}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{num_particles}\n")
            f.write("ITEM: BOX BOUNDS pp pp fs\n")
            f.write(f"0.0 {box_size}\n")
            f.write(f"0.0 {box_size}\n")
            f.write("0.0 1.2\n")
            f.write("ITEM: ATOMS id type x y z vx vy vz radius fx fy fz mass\n")
            df_dump.to_csv(f, sep=" ", header=False, index=False)

    return out_dir.parent


# ---------------------------------------------------------------------------
# Batch Processing across Multiple Seeds
# ---------------------------------------------------------------------------
def run_batch_seeds(
    base_dir: Union[str, Path],
    num_diameters: float = 2.0,
    method: str = "grid",
    shaking_windows: Optional[List[Tuple[float, float]]] = None,
    start_dump: Optional[int] = None,
    end_dump: Optional[int] = None,
    include_bed_prep: bool = False,
    output_log: Optional[Union[str, Path]] = None,
    step: int = 1,
) -> dict:
    """Run top-layer analysis across all seed_* directories under base_dir."""
    base_dir = Path(base_dir)
    seed_dirs = sorted(list(base_dir.glob("seed_*")))

    if not seed_dirs:
        print(f"No seed_* subdirectories found under '{base_dir}'. Checking '{base_dir}' directly.")
        seed_dirs = [base_dir]

    all_seed_results = {}
    combined_summary_rows = []
    combined_comp_rows = []

    for s_dir in seed_dirs:
        print(f"\n>>> Processing: {s_dir.name} ...")
        try:
            res = analyze_bed_shaking_series(
                sim_dir=s_dir,
                num_diameters=num_diameters,
                method=method,
                shaking_windows=shaking_windows,
                start_dump=start_dump,
                end_dump=end_dump,
                include_bed_prep=include_bed_prep,
                output_log=output_log,
                step=step,
                verbose=True,
            )
            out_dir = s_dir / "analysis"
            save_analysis_results(res, output_dir=out_dir)
            all_seed_results[s_dir.name] = res

            summary_df = res["phase_summary_df"].copy()
            summary_df["seed"] = s_dir.name
            combined_summary_rows.append(summary_df)

            comp_df = res.get("shaking_comparison_df", pd.DataFrame())
            if not comp_df.empty:
                comp_df = comp_df.copy()
                comp_df["seed"] = s_dir.name
                combined_comp_rows.append(comp_df)
        except Exception as e:
            print(f"Error processing {s_dir}: {e}")

    if combined_summary_rows:
        df_all = pd.concat(combined_summary_rows, ignore_index=True)
        batch_csv = base_dir / "all_seeds_top_layer_summary.csv"
        df_all.to_csv(batch_csv, index=False)
        print(f"\n[Batch Summary] Combined summary saved to: {batch_csv}")

    if combined_comp_rows:
        df_comp_all = pd.concat(combined_comp_rows, ignore_index=True)
        comp_batch_csv = base_dir / "all_seeds_shaking_comparison.csv"
        df_comp_all.to_csv(comp_batch_csv, index=False)
        print(f"[Batch Summary] Combined shaking comparison saved to: {comp_batch_csv}")

    return all_seed_results


# ---------------------------------------------------------------------------
# Command-Line Interface
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Analyze the top layer (2 particle diameters) of granular bed before, during, and after shaking."
    )
    parser.add_argument(
        "--sim_dir",
        type=str,
        default=None,
        help="Path to single simulation directory (e.g. bed_shaking/seed_1)",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="bed_shaking",
        help="Base directory containing multiple seed_* simulation folders (default: bed_shaking)",
    )
    parser.add_argument(
        "--layer_depth_d",
        type=float,
        default=2.0,
        help="Top layer thickness in particle diameters (default: 2.0)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="grid",
        choices=["grid", "global_percentile", "global_max"],
        help="Surface extraction method (default: grid)",
    )
    parser.add_argument(
        "--start_dump",
        type=int,
        default=None,
        help="Timestep/dump number to start analysis from (default: auto-detected from first dump in output_shaking.txt)",
    )
    parser.add_argument(
        "--end_dump",
        type=int,
        default=None,
        help="Timestep/dump number to end analysis at (default: auto-detected from output_shaking.txt or last dump)",
    )
    parser.add_argument(
        "--output_log",
        type=str,
        default=None,
        help="Path or filename of shaking log file (default: searches for output_shaking.txt)",
    )
    parser.add_argument(
        "--include_bed_prep",
        action="store_true",
        help="Include bed preparation dumps prior to shaking (default: False, ignores dumps < start_dump)",
    )
    parser.add_argument(
        "--shaking_start",
        type=float,
        default=None,
        help="Start time of floor shaking in seconds (default: auto-detect from in.shaking)",
    )
    parser.add_argument(
        "--shaking_end",
        type=float,
        default=None,
        help="End time of floor shaking in seconds (default: auto-detect from in.shaking, typically 4.0 s)",
    )
    parser.add_argument(
        "--shaking_windows",
        type=str,
        default=None,
        help="Comma-separated shaking windows in seconds, e.g. '0:4,9:13' (default: auto-detect from in.shaking*)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Analyze every N-th dump file (default: 1 = all frames)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save figures and CSVs (default: <sim_dir>/analysis)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run self-test with synthetic bed shaking data to verify pipeline",
    )
    parser.add_argument(
        "--demo_multi",
        action="store_true",
        help="Run self-test with synthetic 2-cycle bed shaking data to verify multi-shaking comparison",
    )

    args = parser.parse_args()

    parsed_windows = None
    if args.shaking_windows:
        parsed_windows = []
        for part in args.shaking_windows.split(","):
            s_str, e_str = part.strip().split(":")
            parsed_windows.append((float(s_str), float(e_str)))

    # Self-test / Demo mode
    if args.demo or args.demo_multi:
        n_demo_shakings = 2 if args.demo_multi else 1
        print(f"\n*** RUNNING SYNTHETIC {'MULTI-SHAKING ' if n_demo_shakings > 1 else ''}DEMO TO VERIFY PIPELINE ***\n")
        demo_dir = Path("/tmp/bed_shaking_demo_multi" if n_demo_shakings > 1 else "/tmp/bed_shaking_demo")
        generate_synthetic_shaking_dataset(
            demo_dir,
            num_frames=33 if n_demo_shakings > 1 else 20,
            n_shakings=n_demo_shakings,
        )
        demo_windows = [(0.0, 4.0), (9.0, 13.0)] if n_demo_shakings > 1 else [(0.0, 4.0)]
        results = analyze_bed_shaking_series(
            sim_dir=demo_dir,
            num_diameters=args.layer_depth_d,
            method=args.method,
            shaking_windows=demo_windows,
            include_bed_prep=True,
            verbose=True,
        )
        out_demo = Path("/tmp/bed_shaking_demo_multi_output" if n_demo_shakings > 1 else "/tmp/bed_shaking_demo_output")
        save_analysis_results(results, output_dir=out_demo)
        print("\n*** DEMO COMPLETED SUCCESSFULLY! ***\n")
        return

    # Specific simulation directory
    if args.sim_dir:
        sim_dir = Path(args.sim_dir)
        out_dir = Path(args.output_dir) if args.output_dir else sim_dir / "analysis"
        results = analyze_bed_shaking_series(
            sim_dir=sim_dir,
            num_diameters=args.layer_depth_d,
            method=args.method,
            shaking_start=args.shaking_start,
            shaking_end=args.shaking_end,
            shaking_windows=parsed_windows,
            start_dump=args.start_dump,
            end_dump=args.end_dump,
            include_bed_prep=args.include_bed_prep,
            output_log=args.output_log,
            step=args.step,
            verbose=True,
        )
        save_analysis_results(results, output_dir=out_dir)

    # Batch across all seeds under base_dir
    else:
        base_dir = Path(args.base_dir)
        # Check if base_dir contains results directly or seed_* folders
        seed_dirs = sorted(list(base_dir.glob("seed_*")))
        if seed_dirs:
            run_batch_seeds(
                base_dir=base_dir,
                num_diameters=args.layer_depth_d,
                method=args.method,
                shaking_windows=parsed_windows,
                start_dump=args.start_dump,
                end_dump=args.end_dump,
                include_bed_prep=args.include_bed_prep,
                output_log=args.output_log,
                step=args.step,
            )
        else:
            # Check if base_dir itself is a simulation directory
            out_dir = Path(args.output_dir) if args.output_dir else base_dir / "analysis"
            results = analyze_bed_shaking_series(
                sim_dir=base_dir,
                num_diameters=args.layer_depth_d,
                method=args.method,
                shaking_start=args.shaking_start,
                shaking_end=args.shaking_end,
                shaking_windows=parsed_windows,
                start_dump=args.start_dump,
                end_dump=args.end_dump,
                include_bed_prep=args.include_bed_prep,
                output_log=args.output_log,
                step=args.step,
                verbose=True,
            )
            save_analysis_results(results, output_dir=out_dir)


if __name__ == "__main__":
    main()

