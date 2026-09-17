import os
import re
import glob
from pathlib import Path
import pandas as pd
import numpy as np

# Import dump file readers from physical_params if available, or define fallback
try:
    from physical_params import read_single_dump, read_dump_files, get_dump_file_header
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from physical_params import read_single_dump, read_dump_files, get_dump_file_header


def format_youngs_modulus(val_pa):
    """Format Young's Modulus value with appropriate unit (GPa, MPa, kPa, Pa)."""
    if val_pa is None or np.isnan(val_pa):
        return "N/A"
    val = float(val_pa)
    if val >= 1e9:
        return f"{val / 1e9:.2f} GPa"
    elif val >= 1e6:
        return f"{val / 1e6:.2f} MPa"
    elif val >= 1e3:
        return f"{val / 1e3:.2f} kPa"
    else:
        return f"{val:.2f} Pa"


def extract_particle_properties(sim_path_or_dump_path):
    """
    Extract density, Young's modulus, and Poisson's ratio for each particle type
    from simulation log/output/input files (located in the simulation directory,
    one directory above the dump file location).

    Parameters:
        sim_path_or_dump_path (str or Path): Path to the dump file or simulation directory.

    Returns:
        tuple: (props_dict, props_df)
            props_dict (dict): {type_id: {'density': float, 'youngs_modulus': float, 'poissons_ratio': float}}
            props_df (pd.DataFrame): DataFrame with columns ['type', 'density (kg/m3)', 'youngs_modulus (Pa)', 'poissons_ratio', 'youngs_modulus_str']
    """
    if sim_path_or_dump_path is None:
        return {}, pd.DataFrame()

    p = Path(sim_path_or_dump_path)
    if p.is_file():
        candidates = [p.parent, p.parent.parent]
    elif p.is_dir():
        candidates = [p, p.parent]
    else:
        candidates = [p]

    search_files = []
    for cand in candidates:
        if cand.is_dir():
            search_files.extend(list(cand.glob("log.*")))
            search_files.extend(list(cand.glob("output*.txt")))
            search_files.extend(list(cand.glob("in.*")))

    # Deduplicate preserving order
    unique_files = []
    for f in search_files:
        if f.is_file() and f not in unique_files:
            unique_files.append(f)

    props = {}
    ym_vals = []
    pr_vals = []
    density_map = {}

    for fpath in unique_files:
        try:
            with open(fpath, "r", errors="ignore") as f:
                content = f.read()

            # 1. Young's Modulus
            if not ym_vals:
                ym_matches = re.findall(
                    r"fix\s+\w+\s+all\s+property/global\s+youngsModulus\s+peratomtype\s+([\d\.\s\+eE\-]+)",
                    content,
                )
                for m in ym_matches:
                    tokens = [
                        float(x)
                        for x in m.strip().split()
                        if re.match(r"^[+-]?[\d\.]+(?:[eE][+-]?\d+)?$", x)
                    ]
                    if len(tokens) >= 2:
                        ym_vals = tokens

            # 2. Poisson's Ratio
            if not pr_vals:
                pr_matches = re.findall(
                    r"fix\s+\w+\s+all\s+property/global\s+poissonsRatio\s+peratomtype\s+([\d\.\s\+eE\-]+)",
                    content,
                )
                for m in pr_matches:
                    tokens = [
                        float(x)
                        for x in m.strip().split()
                        if re.match(r"^[+-]?[\d\.]+(?:[eE][+-]?\d+)?$", x)
                    ]
                    if len(tokens) >= 2:
                        pr_vals = tokens

            # 3. Particle templates density
            pts_matches = re.findall(
                r"fix\s+\w+\s+all\s+particletemplate/sphere\s+\S+\s+atom_type\s+(\d+)\s+density\s+constant\s+([\d\.\+eE\-]+)",
                content,
            )
            for at_type, dens in pts_matches:
                density_map[int(at_type)] = float(dens)

            # 4. Direct density variables
            dens_floor = re.findall(r"variable\s+density_floor\s+equal\s+([\d\.\+eE\-]+)", content)
            if dens_floor and 1 not in density_map:
                try:
                    density_map[1] = float(dens_floor[-1])
                except Exception:
                    pass

            dens_wall = re.findall(r"variable\s+density_wall\s+equal\s+([\d\.\+eE\-]+)", content)
            if dens_wall and 5 not in density_map:
                try:
                    density_map[5] = float(dens_wall[-1])
                except Exception:
                    pass

            dens_p_vars = re.findall(
                r"variable\s+density_particle(\d+)\s+equal\s+([\d\.\+eE\-]+)", content
            )
            for p_idx, dens in dens_p_vars:
                t = int(p_idx) + 1  # particle1 -> type 2, particle2 -> type 3, etc.
                if t not in density_map:
                    try:
                        density_map[t] = float(dens)
                    except Exception:
                        pass
        except Exception:
            pass

    max_types = max(
        len(ym_vals),
        len(pr_vals),
        max(density_map.keys()) if density_map else 0,
    )
    for t in range(1, max_types + 1):
        ym = ym_vals[t - 1] if t - 1 < len(ym_vals) else None
        pr = pr_vals[t - 1] if t - 1 < len(pr_vals) else None
        dens = density_map.get(t, None)
        if ym is not None or pr is not None or dens is not None:
            props[t] = {
                "type": t,
                "density (kg/m3)": dens,
                "youngs_modulus (Pa)": ym,
                "poissons_ratio": pr,
                "youngs_modulus_str": format_youngs_modulus(ym),
            }

    df_props = pd.DataFrame(list(props.values()))
    return props, df_props


def analyze_particle_distribution_x(
    dump_input,
    num_parts=10,
    exclude_types=(1, 5),
    x_range=None,
    use_box_bounds=True,
    sim_dir=None,
    verbose=False,
):
    """
    Analyze the percentage and count of each particle type across equal divisions (bins)
    in the x-dimension of the simulation box.

    Parameters:
        dump_input (str, Path, pd.DataFrame, or tuple):
            Path to a .post / .dump file, a loaded DataFrame, or a (metadata, df) tuple.
        num_parts (int):
            Number of equal parts (bins) to divide the x-dimension into. Default is 10.
        exclude_types (tuple or list, optional):
            Particle types to exclude (e.g. 1 for floor, 5 for wall). Default is (1, 5).
        x_range (tuple of float, optional):
            Explicit (x_min, x_max) bounds. If None, determined from box bounds or particle extents.
        use_box_bounds (bool):
            If True and metadata is available, use simulation box bounds for x_min and x_max.
            If False or no metadata, uses the bounding limits of the particles.
        sim_dir (str or Path, optional):
            Directory containing simulation outputs/logs. If None, inferred from dump_input path.
        verbose (bool):
            If True, prints a summary table of the distribution and material properties.

    Returns:
        dict: A dictionary containing:
            - 'timestep': int or None
            - 'num_parts': int
            - 'bin_edges': np.ndarray of shape (num_parts + 1,)
            - 'x_range': tuple (x_min, x_max)
            - 'summary_table': pd.DataFrame (pivoted table with % and counts per bin)
            - 'detailed_df': pd.DataFrame (long-form per-bin, per-type statistics)
            - 'particle_properties': dict of physical properties per atom type
            - 'particle_properties_df': pd.DataFrame of material properties
            - 'total_particles': int
    """
    # 1. Handle dictionary of timesteps (e.g. from read_dump_files)
    if isinstance(dump_input, dict):
        all_results = {}
        for ts, d_input in dump_input.items():
            all_results[ts] = analyze_particle_distribution_x(
                dump_input=d_input,
                num_parts=num_parts,
                exclude_types=exclude_types,
                x_range=x_range,
                use_box_bounds=use_box_bounds,
                sim_dir=sim_dir,
                verbose=False,
            )
        if verbose:
            print(f"Processed {len(all_results)} timesteps for particle distribution along X.")
        return all_results

    # Single dump input
    timestep = None
    box_bounds = None
    file_origin = sim_dir

    if isinstance(dump_input, (str, Path)):
        file_origin = dump_input
        meta, df = read_single_dump(dump_input)
        timestep = meta.get("timestep", None)
        box_bounds = meta.get("box_bounds", None)
    elif isinstance(dump_input, tuple) and len(dump_input) == 2:
        meta, df = dump_input
        if isinstance(meta, dict):
            timestep = meta.get("timestep", None)
            box_bounds = meta.get("box_bounds", None)
    elif isinstance(dump_input, pd.DataFrame):
        df = dump_input.copy()
    else:
        raise TypeError("dump_input must be a file path, DataFrame, (metadata, df) tuple, or dict of timesteps.")

    if "x" not in df.columns or "type" not in df.columns:
        raise ValueError("DataFrame must contain 'x' and 'type' columns.")

    # 2. Extract material properties from one directory above dump file
    props_dict, props_df = extract_particle_properties(file_origin)

    # 3. Filter out excluded particle types
    if exclude_types is not None and len(exclude_types) > 0:
        df_particles = df[~df["type"].isin(exclude_types)].copy()
    else:
        df_particles = df.copy()

    total_particles = len(df_particles)
    if total_particles == 0:
        raise ValueError("No particles found after applying type exclusion filters.")

    # 4. Determine x range
    if x_range is not None:
        x_min, x_max = float(x_range[0]), float(x_range[1])
    elif use_box_bounds and box_bounds is not None:
        x_min, x_max = float(box_bounds[0][0]), float(box_bounds[0][1])
    else:
        has_radius = "radius" in df_particles.columns
        r_offset = df_particles["radius"] if has_radius else 0.0
        x_min = float((df_particles["x"] - r_offset).min())
        x_max = float((df_particles["x"] + r_offset).max())

    if x_min >= x_max:
        raise ValueError(f"Invalid x_range: x_min ({x_min}) >= x_max ({x_max})")

    # 5. Create bins along x
    bin_edges = np.linspace(x_min, x_max, num_parts + 1)
    bin_indices = np.digitize(df_particles["x"], bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_parts - 1)
    df_particles["bin_idx"] = bin_indices

    # 6. Extract unique particle types (sorted)
    all_types = sorted(df_particles["type"].unique())
    has_mass = "mass" in df_particles.columns

    # 7. Compute statistics per bin and per type
    detailed_rows = []
    summary_rows = []
    overall_type_counts = df_particles["type"].value_counts().to_dict()

    for i in range(num_parts):
        b_min = bin_edges[i]
        b_max = bin_edges[i + 1]
        b_center = 0.5 * (b_min + b_max)
        
        bin_df = df_particles[df_particles["bin_idx"] == i]
        bin_total = len(bin_df)
        bin_mass_total = float(bin_df["mass"].sum()) if has_mass else np.nan

        summary_row = {
            "bin": i + 1,
            "x_min": b_min,
            "x_max": b_max,
            "x_center": b_center,
            "total_particles": bin_total,
        }

        for p_type in all_types:
            type_subset = bin_df[bin_df["type"] == p_type]
            count = len(type_subset)
            
            pct_in_bin = (count / bin_total * 100.0) if bin_total > 0 else 0.0
            frac_in_bin = (count / bin_total) if bin_total > 0 else 0.0

            global_type_total = overall_type_counts.get(p_type, 0)
            pct_of_type_global = (count / global_type_total * 100.0) if global_type_total > 0 else 0.0

            row_data = {
                "bin": i + 1,
                "x_min": b_min,
                "x_max": b_max,
                "x_center": b_center,
                "type": int(p_type),
                "count": count,
                "total_in_bin": bin_total,
                "pct_in_bin": pct_in_bin,
                "frac_in_bin": frac_in_bin,
                "pct_of_type_global": pct_of_type_global,
            }

            if has_mass:
                p_mass = float(type_subset["mass"].sum())
                mass_pct_in_bin = (p_mass / bin_mass_total * 100.0) if (bin_mass_total > 0) else 0.0
                row_data["mass (kg)"] = p_mass
                row_data["mass_pct_in_bin"] = mass_pct_in_bin
                summary_row[f"type_{p_type}_mass_pct"] = mass_pct_in_bin

            detailed_rows.append(row_data)

            # Summary table columns
            summary_row[f"type_{p_type}_count"] = count
            summary_row[f"type_{p_type}_pct"] = pct_in_bin

        summary_rows.append(summary_row)

    df_detailed = pd.DataFrame(detailed_rows)
    df_summary = pd.DataFrame(summary_rows)

    results = {
        "timestep": timestep,
        "num_parts": num_parts,
        "bin_edges": bin_edges,
        "x_range": (x_min, x_max),
        "total_particles": total_particles,
        "summary_table": df_summary,
        "detailed_df": df_detailed,
        "particle_properties": props_dict,
        "particle_properties_df": props_df,
    }

    if verbose:
        print("=" * 70)
        title = f"PARTICLE TYPE DISTRIBUTION ALONG X ({num_parts} PARTS)"
        if timestep is not None:
            title += f" [Timestep: {timestep}]"
        print(title)
        print("=" * 70)
        print(f"X Range: [{x_min:.4f}, {x_max:.4f}] m | Bin Width: {(x_max - x_min) / num_parts:.4f} m")
        print(f"Total Particles (excluding types {exclude_types}): {total_particles:,}")
        print("-" * 70)
        
        display_cols = ["bin", "x_min", "x_max", "total_particles"]
        for p_type in all_types:
            display_cols.append(f"type_{p_type}_pct")
            display_cols.append(f"type_{p_type}_count")
        
        print(df_summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        print("=" * 70)

        if not props_df.empty:
            print("\n" + "=" * 70)
            print("                 PARTICLE MATERIAL PROPERTIES")
            print("=" * 70)
            print(props_df.to_string(index=False))
            print("=" * 70 + "\n")

    return results


def plot_particle_distribution_x(
    results,
    mode="percentage",
    show_properties_box=True,
    properties_loc="upper right",
    save_path=None,
    show=True,
):
    """
    Plot the particle distribution across x bins with an optional material properties box.

    Parameters:
        results (dict): Output from analyze_particle_distribution_x.
        mode (str):
            - 'percentage': Stacked bar chart showing percentage composition in each bin.
            - 'count': Grouped bar chart showing counts of each type per bin.
            - 'lines': Line plot of type percentage vs x_center.
        show_properties_box (bool):
            Whether to display density, Young's modulus, and Poisson's ratio in a separate box.
        properties_loc (str):
            Placement of property box ('upper right', 'upper left', 'side', 'outside').
        save_path (str or Path, optional): Path to save the figure.
        show (bool): Whether to display the plot.
    """
    import matplotlib.pyplot as plt

    summary_df = results["summary_table"]
    num_parts = results["num_parts"]
    timestep = results.get("timestep", None)
    props_dict = results.get("particle_properties", {})
    
    # Identify type columns
    pct_cols = [c for c in summary_df.columns if c.endswith("_pct") and not c.endswith("_mass_pct")]
    count_cols = [c for c in summary_df.columns if c.endswith("_count")]
    types = [int(re.search(r"type_(\d+)_", c).group(1)) for c in pct_cols]

    figsize = (12, 6) if (show_properties_box and properties_loc in ["side", "outside"]) else (10, 6)
    fig, ax = plt.subplots(figsize=figsize)

    if mode == "percentage":
        bottom = np.zeros(len(summary_df))
        bin_labels = [f"[{row.x_min:.2f}, {row.x_max:.2f}]" for _, row in summary_df.iterrows()]
        x_positions = np.arange(len(summary_df))

        for p_type, col in zip(types, pct_cols):
            values = summary_df[col].values
            ax.bar(x_positions, values, bottom=bottom, label=f"Type {p_type}", alpha=0.85, edgecolor="black", linewidth=0.5)
            bottom += values

        ax.set_xticks(x_positions)
        ax.set_xticklabels(bin_labels, rotation=45, ha="right")
        ax.set_ylabel("Percentage in Bin (%)", fontsize=12)
        ax.set_ylim(0, 100)
        ax.set_title(f"Particle Type Percentage Distribution along X ({num_parts} parts)" + (f" - Timestep {timestep}" if timestep else ""), fontsize=13, fontweight="bold")
        ax.legend(title="Particle Type", loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    elif mode == "count":
        x_positions = np.arange(len(summary_df))
        width = 0.8 / len(types)
        bin_labels = [f"[{row.x_min:.2f}, {row.x_max:.2f}]" for _, row in summary_df.iterrows()]

        for i, (p_type, col) in enumerate(zip(types, count_cols)):
            offset = (i - len(types) / 2 + 0.5) * width
            ax.bar(x_positions + offset, summary_df[col].values, width=width, label=f"Type {p_type}", edgecolor="black", linewidth=0.5)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(bin_labels, rotation=45, ha="right")
        ax.set_ylabel("Particle Count", fontsize=12)
        ax.set_title(f"Particle Count per Type along X ({num_parts} parts)" + (f" - Timestep {timestep}" if timestep else ""), fontsize=13, fontweight="bold")
        ax.legend(title="Particle Type", loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    elif mode == "lines":
        x_centers = summary_df["x_center"].values
        for p_type, col in zip(types, pct_cols):
            ax.plot(x_centers, summary_df[col].values, marker="o", linewidth=2, label=f"Type {p_type}")

        ax.set_xlabel("X Position (m)", fontsize=12)
        ax.set_ylabel("Percentage in Bin (%)", fontsize=12)
        ax.set_title(f"Particle Type Percentage Profile along X" + (f" - Timestep {timestep}" if timestep else ""), fontsize=13, fontweight="bold")
        ax.legend(title="Particle Type", loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.5)

    # Add separate Material Properties Info Box
    if show_properties_box and props_dict:
        prop_lines = ["Material Properties:"]
        for p_type in types:
            p_info = props_dict.get(p_type, {})
            dens = p_info.get("density (kg/m3)", "N/A")
            ym = p_info.get("youngs_modulus_str", format_youngs_modulus(p_info.get("youngs_modulus (Pa)")))
            pr = p_info.get("poissons_ratio", "N/A")

            dens_str = f"{dens:.0f} kg/m³" if isinstance(dens, (int, float)) else str(dens)
            pr_str = f"{pr:.2f}" if isinstance(pr, (int, float)) else str(pr)

            prop_lines.append(f"Type {p_type}:  ρ={dens_str} | E={ym} | ν={pr_str}")

        prop_text = "\n".join(prop_lines)

        bbox_props = dict(
            boxstyle="round,pad=0.6",
            facecolor="#f8f9fa",
            edgecolor="#495057",
            linewidth=1.2,
            alpha=0.92,
        )

        if properties_loc == "side" or properties_loc == "outside":
            ax.text(
                1.02,
                0.5,
                prop_text,
                transform=ax.transAxes,
                fontsize=9.5,
                verticalalignment="center",
                bbox=bbox_props,
                fontfamily="monospace",
            )
        elif properties_loc == "upper right":
            ax.text(
                0.98,
                0.96,
                prop_text,
                transform=ax.transAxes,
                fontsize=9.5,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=bbox_props,
                fontfamily="monospace",
            )
        elif properties_loc == "upper left":
            ax.text(
                0.02,
                0.96,
                prop_text,
                transform=ax.transAxes,
                fontsize=9.5,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=bbox_props,
                fontfamily="monospace",
            )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze percentage of particle types along X dimension.")
    parser.add_argument("--dump_file", type=str, default="/Volumes/Sandisk/post-doc/incline_avalanche/seed_1/results/dump10245000.post", help="Path to dump file.")
    parser.add_argument("--num_parts", type=int, default=10, help="Number of equal divisions in X dimension.")
    parser.add_argument("--exclude_types", type=int, nargs="*", default=[1, 5], help="Particle types to exclude (e.g. 1 5).")
    parser.add_argument("--mode", type=str, default="percentage", choices=["percentage", "count", "lines"], help="Plotting mode.")
    parser.add_argument("--plot", action="store_true", help="Display plot.")
    parser.add_argument("--save_plot", type=str, default=None, help="Filepath to save plot image.")

    args = parser.parse_args()

    if os.path.exists(args.dump_file):
        res = analyze_particle_distribution_x(
            dump_input=args.dump_file,
            num_parts=args.num_parts,
            exclude_types=args.exclude_types,
            verbose=True,
        )
        if args.plot or args.save_plot:
            plot_particle_distribution_x(res, mode=args.mode, save_path=args.save_plot, show=args.plot)
    else:
        print(f"Dump file '{args.dump_file}' not found. Please provide a valid file with --dump_file.")
