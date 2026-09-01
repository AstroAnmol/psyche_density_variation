import os
import re
import glob
from pathlib import Path
import pandas as pd
import numpy as np


def get_dump_file_header(sim_folder_path, dump_file_path=None):
    """
    Find column names for dump files.
    
    First looks for the `dump ... custom` line in output.txt, log.lammps,
    log.liggghts, or in.* scripts in sim_folder_path.
    Falls back to reading the `ITEM: ATOMS ...` header directly from a dump file.
    
    Parameters:
        sim_folder_path (str or Path): Path to the simulation directory.
        dump_file_path (str or Path, optional): Path to a sample dump file.
        
    Returns:
        list of str: Column names.
    """
    sim_path = Path(sim_folder_path)
    search_files = [
        sim_path / "output.txt",
        sim_path / "log.lammps",
        sim_path / "log.liggghts",
        *sim_path.glob("in.*"),
    ]
    
    for fpath in search_files:
        if fpath.is_file():
            try:
                with open(fpath, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("dump") and "custom" in line:
                            tokens = line.split()
                            try:
                                custom_idx = tokens.index("custom")
                                # Format: dump <id> <group> custom <dumpstep> <file_pattern> <col1> <col2> ...
                                columns = tokens[custom_idx + 3:]
                                if columns:
                                    return columns
                            except (ValueError, IndexError):
                                continue
            except Exception:
                pass
                
    # Fallback: extract directly from the dump file's ITEM: ATOMS header
    if dump_file_path and os.path.isfile(dump_file_path):
        with open(dump_file_path, "r") as f:
            for line in f:
                if line.startswith("ITEM: ATOMS"):
                    return line.replace("ITEM: ATOMS", "").strip().split()
                    
    return None


def read_single_dump(file_path, columns=None):
    """
    Read a single LAMMPS/LIGGGHTS custom dump file.
    
    Parameters:
        file_path (str or Path): Path to the dump file.
        columns (list of str, optional): Explicit column names. If None, read from ITEM: ATOMS.
        
    Returns:
        metadata (dict): Dictionary with 'timestep', 'num_atoms', 'box_bounds'.
        df (pd.DataFrame): DataFrame containing particle/atom data.
    """
    file_path = Path(file_path)
    metadata = {}
    header_cols = columns
    
    with open(file_path, "r") as f:
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
                if header_cols is None:
                    header_cols = line_str.replace("ITEM: ATOMS", "").strip().split()
                break

    if header_cols is None:
        raise ValueError(f"Could not determine column names for dump file: {file_path}")

    # Read data block skipping the 9 header lines
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        skiprows=9,
        names=header_cols,
        engine="c",
    )
    
    return metadata, df


def read_dump_files(sim_folder_path, timesteps=None, return_metadata=False):
    """
    Read all (or selected) dump files from a simulation folder.
    
    Parameters:
        sim_folder_path (str or Path): Path to the simulation directory (e.g. 'trials/setup')
                                       or its 'results' directory.
        timesteps (list or int, optional): Specific timestep(s) to load. If None, loads all.
        return_metadata (bool): If True, returns a dict of {timestep: (metadata, df)}.
                                If False (default), returns {timestep: df}.
                                
    Returns:
        dict: {timestep: df} or {timestep: (metadata, df)} sorted chronologically.
    """
    sim_path = Path(sim_folder_path)
    
    # Check potential locations for dump files
    dump_patterns = [
        sim_path / "results" / "dump*.post",
        sim_path / "results" / "dump*",
        sim_path / "dump*.post",
        sim_path / "dump*",
    ]
    
    dump_files = []
    for pattern in dump_patterns:
        matched = glob.glob(str(pattern))
        if matched:
            dump_files = matched
            break
            
    if not dump_files:
        raise FileNotFoundError(f"No dump files found in '{sim_folder_path}' or '{sim_folder_path}/results'")
        
    def extract_timestep(f):
        match = re.search(r"dump(\d+)", os.path.basename(f))
        return int(match.group(1)) if match else -1
        
    # Map and sort numerically by timestep
    file_dict = {extract_timestep(f): f for f in dump_files if extract_timestep(f) >= 0}
    sorted_timesteps = sorted(file_dict.keys())
    
    if timesteps is not None:
        if isinstance(timesteps, (int, np.integer)):
            timesteps = [timesteps]
        sorted_timesteps = [ts for ts in sorted_timesteps if ts in timesteps]
        
    sample_file = file_dict[sorted_timesteps[0]] if sorted_timesteps else dump_files[0]
    header = get_dump_file_header(sim_folder_path, dump_file_path=sample_file)
    
    results = {}
    for ts in sorted_timesteps:
        fpath = file_dict[ts]
        meta, df = read_single_dump(fpath, columns=header)
        results[ts] = (meta, df) if return_metadata else df
        
    return results


# Backward compatibility / alias
read_damp_files = read_dump_files

def physical_params(dump_input, exclude_types=(4, 5), custom_volume=None, verbose=False):
    """
    Given a dump file path or DataFrame, compute and return the key physical parameters:
      - Bulk density & Solid (grain) density
      - Total number of particles & distribution by type (number & mass fractions)
      - Packing fraction & Porosity
      - Total mass & solid volume
      - Bounding box dimensions & Center of Mass

    Parameters:
        dump_input (str, Path, pd.DataFrame, or tuple):
            Path to a .post / .dump file, a loaded DataFrame, or a (metadata, df) tuple.
        exclude_types (tuple or list):
            Particle types to exclude (e.g. 4 for floor, 5 for wall). Default is (4, 5).
        custom_volume (float, optional):
            Custom bed volume in m^3. If None, computed from the bounding box of particle extents.
        verbose (bool):
            If True, prints a formatted summary of physical parameters.

    Returns:
        dict: Dictionary containing:
            - 'timestep': int
            - 'total_particles': int
            - 'total_mass': float (kg)
            - 'solid_volume': float (m^3)
            - 'bed_volume': float (m^3)
            - 'bulk_density': float (kg/m^3)
            - 'solid_density': float (kg/m^3)
            - 'packing_fraction': float (solid_vol / bed_vol)
            - 'porosity': float (1 - packing_fraction)
            - 'center_of_mass': np.ndarray [x, y, z]
            - 'bed_bounds': dict with bounding box limits & dimensions
            - 'by_type': pd.DataFrame with per-type breakdown (count, num_fraction, mass, mass_fraction, vol_fraction)
    """
    # 1. Handle various input types
    timestep = None
    if isinstance(dump_input, (str, Path)):
        meta, df = read_single_dump(dump_input)
        timestep = meta.get("timestep", None)
    elif isinstance(dump_input, tuple) and len(dump_input) == 2:
        meta, df = dump_input
        timestep = meta.get("timestep", None) if isinstance(meta, dict) else None
    elif isinstance(dump_input, pd.DataFrame):
        df = dump_input
    else:
        raise TypeError("dump_input must be a file path, DataFrame, or (metadata, df) tuple.")

    # 2. Filter out non-granular / boundary types (e.g. floor=4, wall=5)
    if exclude_types and "type" in df.columns:
        df_particles = df[~df["type"].isin(exclude_types)].copy()
    else:
        df_particles = df.copy()

    total_particles = len(df_particles)
    if total_particles == 0:
        raise ValueError("No particles found after applying type filters.")

    # 3. Mass & Solid Volume calculation
    has_radius = "radius" in df_particles.columns
    has_mass = "mass" in df_particles.columns

    if has_radius:
        particle_volumes = (4.0 / 3.0) * np.pi * (df_particles["radius"] ** 3)
        solid_volume = float(particle_volumes.sum())
    else:
        particle_volumes = None
        solid_volume = np.nan

    if has_mass:
        total_mass = float(df_particles["mass"].sum())
    else:
        total_mass = np.nan

    # Solid (grain) density
    solid_density = (total_mass / solid_volume) if (not np.isnan(total_mass) and solid_volume > 0) else np.nan

    # 4. Bed Dimensions & Bounding Volume
    r_offset = df_particles["radius"] if has_radius else 0.0
    x_min = float((df_particles["x"] - r_offset).min())
    x_max = float((df_particles["x"] + r_offset).max())
    y_min = float((df_particles["y"] - r_offset).min())
    y_max = float((df_particles["y"] + r_offset).max())
    z_min = float((df_particles["z"] - r_offset).min())
    z_max = float((df_particles["z"] + r_offset).max())

    lx = max(x_max - x_min, 0.0)
    ly = max(y_max - y_min, 0.0)
    lz = max(z_max - z_min, 0.0)

    if custom_volume is not None and custom_volume > 0:
        bed_volume = float(custom_volume)
    else:
        bed_volume = float(lx * ly * lz)

    # 5. Bulk Density, Packing Fraction & Porosity
    bulk_density = (total_mass / bed_volume) if (not np.isnan(total_mass) and bed_volume > 0) else np.nan
    packing_fraction = (solid_volume / bed_volume) if (not np.isnan(solid_volume) and bed_volume > 0) else np.nan
    porosity = (1.0 - packing_fraction) if not np.isnan(packing_fraction) else np.nan

    # 6. Center of Mass
    if has_mass and total_mass > 0:
        com_x = float((df_particles["x"] * df_particles["mass"]).sum() / total_mass)
        com_y = float((df_particles["y"] * df_particles["mass"]).sum() / total_mass)
        com_z = float((df_particles["z"] * df_particles["mass"]).sum() / total_mass)
    else:
        com_x = float(df_particles["x"].mean())
        com_y = float(df_particles["y"].mean())
        com_z = float(df_particles["z"].mean())
    com = np.array([com_x, com_y, com_z])

    # 7. Distribution by Particle Type
    type_stats = []
    if "type" in df_particles.columns:
        for p_type, group in df_particles.groupby("type"):
            count = len(group)
            num_frac = count / total_particles
            
            p_mass = float(group["mass"].sum()) if has_mass else np.nan
            mass_frac = (p_mass / total_mass) if (has_mass and total_mass > 0) else np.nan
            
            if has_radius:
                p_vol = float(((4.0 / 3.0) * np.pi * (group["radius"] ** 3)).sum())
                vol_frac = (p_vol / solid_volume) if solid_volume > 0 else np.nan
                mean_dia = float((group["radius"] * 2).mean())
            else:
                p_vol = np.nan
                vol_frac = np.nan
                mean_dia = np.nan

            type_stats.append({
                "type": int(p_type),
                "count": count,
                "num_fraction": num_frac,
                "mass (kg)": p_mass,
                "mass_fraction": mass_frac,
                "volume (m^3)": p_vol,
                "vol_fraction": vol_frac,
                "mean_diameter (m)": mean_dia,
            })
    df_by_type = pd.DataFrame(type_stats)

    results = {
        "timestep": timestep,
        "total_particles": total_particles,
        "total_mass": total_mass,
        "solid_volume": solid_volume,
        "bed_volume": bed_volume,
        "bulk_density": bulk_density,
        "solid_density": solid_density,
        "packing_fraction": packing_fraction,
        "porosity": porosity,
        "center_of_mass": com,
        "bed_bounds": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "z_min": z_min,
            "z_max": z_max,
            "length_x": lx,
            "width_y": ly,
            "height_z": lz,
        },
        "by_type": df_by_type,
    }

    if verbose:
        print("=" * 55)
        print(f"  PHYSICAL PARAMETERS SUMMARY" + (f" (Timestep: {timestep})" if timestep else ""))
        print("=" * 55)
        print(f"Total Particles   : {total_particles:,}")
        print(f"Total Mass        : {total_mass:.4e} kg")
        print(f"Solid Volume      : {solid_volume:.4e} m^3")
        print(f"Bed Volume        : {bed_volume:.4e} m^3")
        print(f"Bulk Density      : {bulk_density:.2f} kg/m^3")
        print(f"Grain Density     : {solid_density:.2f} kg/m^3")
        print(f"Packing Fraction  : {packing_fraction:.4f}")
        print(f"Porosity          : {porosity:.4f}")
        print(f"Center of Mass    : [x={com[0]:.4f}, y={com[1]:.4f}, z={com[2]:.4f}] m")
        print(f"Bed Dimensions    : {lx:.3f} m (L) x {ly:.3f} m (W) x {lz:.3f} m (H)")
        print("\n--- Particle Type Breakdown ---")
        if not df_by_type.empty:
            print(df_by_type.to_string(index=False))
        print("=" * 55)

    return results

physical_params(dump_input="trials/setup/results/dump2500000.post", exclude_types=[4, 5], verbose=True)