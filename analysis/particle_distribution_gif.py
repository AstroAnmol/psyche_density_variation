"""
particle_distribution_gif.py
============================
Generate an animated GIF showing how the particle-type distribution along the
X-axis evolves over time, using the LIGGGHTS / LAMMPS dump files found in the
`./results` subfolder of a given simulation directory.

Frame ordering is determined by the integer frame-number embedded in each dump
filename (`dump<N>.post` or `dump<N>`). Physical simulation time is derived from that
frame number multiplied by the timestep `dt` read from the log / input files
in the parent directory or simulation folder.

Usage
-----
python particle_distribution_gif.py --sim_dir /path/to/sim_dir [OPTIONS]

Options
-------
  --sim_dir        Path to simulation directory (must contain a `results/`
                   subdirectory with dump files and log/input/output files in
                   the parent directory or sim_dir itself).
  --num_parts      Number of equal x-bins for the distribution (default: 10).
  --exclude_types  Particle types to exclude, space-separated (default: 1 5).
  --step           Use every N-th dump file, sorted chronologically
                   (default: 1 = all files).
  --fps            Frames per second in the output GIF (default: 5).
  --output         Path for the output GIF file
                   (default: <sim_dir>/particle_distribution.gif).
  --dpi            Figure DPI for each frame (default: 120).
  --mode           Plot mode: percentage | count | lines | delta_percentage (or delta_pct/change) | delta_lines
                   (default: percentage).
                   'delta_percentage': grouped bar chart showing change in percentage (Δ%)
                   relative to the first frame (t0) for each particle type in each bin.
                   'delta_lines': line plot showing Δ% vs X bin centers.
  --no_props       Suppress material-properties box on each frame.

Dependencies
------------
  matplotlib, Pillow (PIL), numpy, pandas
  physical_params.py  (read_single_dump, read_dump_files, get_dump_file_header)
  particle_distribution.py  (analyze_particle_distribution_x,
                              extract_particle_properties,
                              format_youngs_modulus)

Examples
--------
# All frames, 5 fps
python particle_distribution_gif.py --sim_dir /Volumes/Sandisk/post-doc/incline_shaking/seed_1

# Change in percentage (Δ%) relative to initial frame t0
python particle_distribution_gif.py \\
    --sim_dir /Volumes/Sandisk/post-doc/incline_shaking/seed_1 \\
    --mode delta_percentage --step 5 --fps 8

# Every 10th frame, 8 fps, save to a custom path
python particle_distribution_gif.py \\
    --sim_dir /Volumes/Sandisk/post-doc/incline_shaking/seed_1 \\
    --step 10 --fps 8 --output /tmp/shaking_seed1.gif
"""

import os
import re
import glob
import argparse
import io
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend safe for GIF rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    from PIL import Image
except ImportError:
    raise ImportError(
        "Pillow is required to create GIFs. Install it with: pip install Pillow"
    )

# ---------------------------------------------------------------------------
# Import shared utilities from the analysis package
# ---------------------------------------------------------------------------
try:
    from physical_params import read_single_dump, get_dump_file_header
    from particle_distribution import (
        analyze_particle_distribution_x,
        extract_particle_properties,
        format_youngs_modulus,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from physical_params import read_single_dump, get_dump_file_header
    from particle_distribution import (
        analyze_particle_distribution_x,
        extract_particle_properties,
        format_youngs_modulus,
    )


# ---------------------------------------------------------------------------
# Timestep (dt) extraction from log / input / output files
# ---------------------------------------------------------------------------

def extract_dt_from_files(sim_dir: Path) -> float | None:
    """
    Search log.*, output*.txt, and in.* files in sim_dir and its parent
    for the simulation timestep `dt`.

    Handles LIGGGHTS / LAMMPS `timestep <value>` commands.

    Returns
    -------
    float or None
        The timestep in seconds, or None if not found.
    """
    candidates = []
    for pattern in ("log.*", "output*.txt", "in.*"):
        candidates.extend(sim_dir.glob(pattern))
    
    # Also check parent directory
    parent = sim_dir.parent
    if parent != sim_dir:
        for pattern in ("log.*", "output*.txt", "in.*"):
            candidates.extend(parent.glob(pattern))

    seen = set()
    unique = []
    for f in candidates:
        if f.is_file() and f not in seen:
            seen.add(f)
            unique.append(f)

    for fpath in unique:
        try:
            with open(fpath, "r", errors="ignore") as fh:
                for line in fh:
                    m = re.match(
                        r"^\s*timestep\s+([+\-]?[\d\.]+(?:[eE][+\-]?\d+)?)\s*$",
                        line,
                        re.IGNORECASE,
                    )
                    if m:
                        return float(m.group(1))
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Dump-file discovery
# ---------------------------------------------------------------------------

def discover_dump_files(sim_dir: Path) -> list[tuple[int, Path]]:
    """
    Find all dump<N>.post (or dump<N>) files inside <sim_dir>/results/.

    Returns
    -------
    list of (frame_number, Path) sorted by frame_number ascending.
    """
    results_dir = sim_dir / "results"
    if not results_dir.is_dir():
        # Fallback to sim_dir directly if results doesn't exist
        if (sim_dir / "dump0.post").exists() or list(sim_dir.glob("dump*.post")):
            results_dir = sim_dir
        else:
            raise FileNotFoundError(
                f"No 'results' subdirectory found inside '{sim_dir}'."
            )

    patterns = ["dump*.post", "dump*"]
    matched: list[Path] = []
    for pat in patterns:
        found = [
            p for p in results_dir.glob(pat)
            if p.is_file() and not p.name.startswith("._")  # skip macOS metadata
        ]
        if found:
            matched = found
            break

    if not matched:
        raise FileNotFoundError(
            f"No dump files found in '{results_dir}'."
        )

    def extract_frame_num(p: Path) -> int:
        m = re.search(r"dump(\d+)", p.name)
        return int(m.group(1)) if m else -1

    pairs = [(n, p) for p in matched if (n := extract_frame_num(p)) >= 0]
    pairs.sort(key=lambda x: x[0])
    return pairs


# ---------------------------------------------------------------------------
# Single-frame figure renderer
# ---------------------------------------------------------------------------

def render_frame(
    results: dict,
    frame_number: int,
    physical_time: float | None,
    dt: float | None,
    mode: str,
    show_props: bool,
    dpi: int,
    global_y_max: float,
    frame_index: int,
    total_frames: int,
    color_map: dict,
    ref_results: dict | None = None,
) -> Image.Image:
    """
    Render one animation frame and return it as a PIL Image.
    """
    summary_df = results["summary_table"]
    num_parts = results["num_parts"]
    props_dict = results.get("particle_properties", {})

    pct_cols = [c for c in summary_df.columns if c.endswith("_pct") and not c.endswith("_mass_pct")]
    count_cols = [c for c in summary_df.columns if c.endswith("_count")]
    types = [int(re.search(r"type_(\d+)_", c).group(1)) for c in pct_cols]

    is_delta = mode in ("delta_percentage", "delta_pct", "change", "delta_lines")
    ref_summary_df = ref_results["summary_table"] if (is_delta and ref_results) else None

    # Layout
    fig_w = 12.0 if (show_props and props_dict) else 10.0
    fig, ax = plt.subplots(figsize=(fig_w, 5.5), dpi=dpi)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    spine_color = "#a8b2d8"
    for spine in ax.spines.values():
        spine.set_edgecolor(spine_color)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=spine_color, labelsize=8)

    x_positions = np.arange(len(summary_df))
    bin_labels = [f"{row.x_center:.3f}" for _, row in summary_df.iterrows()]

    if mode == "percentage":
        bottom = np.zeros(len(summary_df))
        for p_type, col in zip(types, pct_cols):
            values = summary_df[col].values
            ax.bar(
                x_positions,
                values,
                bottom=bottom,
                label=f"Type {p_type}",
                color=color_map.get(p_type),
                alpha=0.88,
                edgecolor="#ffffff",
                linewidth=0.3,
            )
            bottom += values

        ax.set_ylabel("Percentage in Bin (%)", color=spine_color, fontsize=9)
        ax.set_ylim(0, global_y_max * 1.05)

    elif mode == "count":
        width = 0.8 / max(len(types), 1)
        for i, (p_type, col) in enumerate(zip(types, count_cols)):
            offset = (i - len(types) / 2 + 0.5) * width
            ax.bar(
                x_positions + offset,
                summary_df[col].values,
                width=width,
                label=f"Type {p_type}",
                color=color_map.get(p_type),
                edgecolor="#ffffff",
                linewidth=0.3,
            )

        ax.set_ylabel("Particle Count", color=spine_color, fontsize=9)
        ax.set_ylim(0, global_y_max * 1.05)

    elif mode in ("delta_percentage", "delta_pct", "change"):
        width = 0.8 / max(len(types), 1)
        for i, (p_type, col) in enumerate(zip(types, pct_cols)):
            offset = (i - len(types) / 2 + 0.5) * width
            if ref_summary_df is not None and col in ref_summary_df.columns:
                delta_vals = summary_df[col].values - ref_summary_df[col].values
            else:
                delta_vals = np.zeros(len(summary_df))
            ax.bar(
                x_positions + offset,
                delta_vals,
                width=width,
                label=f"Type {p_type}",
                color=color_map.get(p_type),
                edgecolor="#ffffff",
                linewidth=0.3,
            )

        ax.axhline(0, color=spine_color, linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_ylabel("Change in Percentage (Δ%)", color=spine_color, fontsize=9)
        ax.set_ylim(-global_y_max * 1.08, global_y_max * 1.08)

    elif mode == "lines":
        x_centers = summary_df["x_center"].values
        for p_type, col in zip(types, pct_cols):
            ax.plot(
                x_centers,
                summary_df[col].values,
                marker="o",
                markersize=4,
                linewidth=2,
                label=f"Type {p_type}",
                color=color_map.get(p_type),
            )
        ax.set_xlabel("X Position (m)", color=spine_color, fontsize=9)
        ax.set_ylabel("Percentage in Bin (%)", color=spine_color, fontsize=9)
        ax.set_ylim(0, global_y_max * 1.05)

    elif mode == "delta_lines":
        x_centers = summary_df["x_center"].values
        for p_type, col in zip(types, pct_cols):
            if ref_summary_df is not None and col in ref_summary_df.columns:
                delta_vals = summary_df[col].values - ref_summary_df[col].values
            else:
                delta_vals = np.zeros(len(summary_df))
            ax.plot(
                x_centers,
                delta_vals,
                marker="o",
                markersize=4,
                linewidth=2,
                label=f"Type {p_type}",
                color=color_map.get(p_type),
            )
        ax.axhline(0, color=spine_color, linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("X Position (m)", color=spine_color, fontsize=9)
        ax.set_ylabel("Change in Percentage (Δ%)", color=spine_color, fontsize=9)
        ax.set_ylim(-global_y_max * 1.08, global_y_max * 1.08)

    if mode in ("percentage", "count", "delta_percentage", "delta_pct", "change"):
        ax.set_xticks(x_positions)
        ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=7, color=spine_color)
        ax.set_xlabel("X Bin Center (m)", color=spine_color, fontsize=9)

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="y", linestyle="--", alpha=0.3, color=spine_color)

    # Title & simulation time
    if physical_time is not None:
        time_str = f"Time t = {physical_time:.4f} s (frame {frame_number})"
    else:
        time_str = f"Frame {frame_number}"

    if is_delta and frame_index == 0:
        time_str += " [Initial Reference t₀]"

    mode_label = {
        "percentage": "% Distribution",
        "count": "Count",
        "lines": "% Profile",
        "delta_percentage": "Δ% Change (vs t₀)",
        "delta_pct": "Δ% Change (vs t₀)",
        "change": "Δ% Change (vs t₀)",
        "delta_lines": "Δ% Profile (vs t₀)",
    }.get(mode, "% Distribution")
    title = f"Particle Type {mode_label} along X [{num_parts} bins]"
    ax.set_title(title, color="#e2e8f0", fontsize=11, fontweight="bold", pad=8)

    # Subtitle for physical time
    fig.text(
        0.5,
        0.94,
        time_str,
        ha="center",
        va="top",
        fontsize=10,
        color="#63b3ed",
        fontweight="semibold",
    )

    # Frame progress counter
    fig.text(
        0.98,
        0.97,
        f"{frame_index + 1}/{total_frames}",
        ha="right",
        va="top",
        fontsize=7,
        color="#718096",
    )

    # Legend
    leg = ax.legend(
        title="Particle Type",
        loc="upper right",
        fontsize=8,
        title_fontsize=8,
        framealpha=0.35,
        facecolor="#1a1a2e",
        edgecolor=spine_color,
        labelcolor="#e2e8f0",
    )
    if leg:
        leg.get_title().set_color("#e2e8f0")

    # Material properties box
    if show_props and props_dict:
        prop_lines = ["Material Properties"]
        for p_type in types:
            p_info = props_dict.get(p_type, {})
            dens = p_info.get("density (kg/m3)", "N/A")
            ym = p_info.get("youngs_modulus_str", format_youngs_modulus(p_info.get("youngs_modulus (Pa)")))
            pr = p_info.get("poissons_ratio", "N/A")
            dens_str = f"{dens:.0f} kg/m³" if isinstance(dens, (int, float)) else str(dens)
            pr_str = f"{pr:.2f}" if isinstance(pr, (int, float)) else str(pr)
            prop_lines.append(f"Type {p_type}: ρ={dens_str} | E={ym} | ν={pr_str}")

        bbox_props = dict(
            boxstyle="round,pad=0.5",
            facecolor="#0d2137",
            edgecolor="#4a90d9",
            linewidth=1.1,
            alpha=0.90,
        )
        ax.text(
            1.02,
            0.5,
            "\n".join(prop_lines),
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="center",
            bbox=bbox_props,
            fontfamily="monospace",
            color="#ccd6f6",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ---------------------------------------------------------------------------
# Global axis limit & colors
# ---------------------------------------------------------------------------

def compute_global_y_max(all_results: list[dict], mode: str, ref_results: dict | None = None) -> float:
    max_val = 0.0
    is_delta = mode in ("delta_percentage", "delta_pct", "change", "delta_lines")
    ref_df = ref_results["summary_table"] if (is_delta and ref_results) else None

    for res in all_results:
        df = res["summary_table"]
        if is_delta:
            pct_cols = [c for c in df.columns if c.endswith("_pct") and not c.endswith("_mass_pct")]
            if pct_cols and ref_df is not None:
                common_cols = [c for c in pct_cols if c in ref_df.columns]
                if common_cols:
                    diff = np.abs(df[common_cols].values - ref_df[common_cols].values)
                    max_val = max(max_val, float(np.nanmax(diff)))
        elif mode == "percentage":
            pct_cols = [c for c in df.columns if c.endswith("_pct") and not c.endswith("_mass_pct")]
            if pct_cols:
                stacked = df[pct_cols].sum(axis=1).max()
                max_val = max(max_val, stacked)
        elif mode == "count":
            cnt_cols = [c for c in df.columns if c.endswith("_count")]
            if cnt_cols:
                max_val = max(max_val, df[cnt_cols].max().max())
        elif mode == "lines":
            pct_cols = [c for c in df.columns if c.endswith("_pct") and not c.endswith("_mass_pct")]
            if pct_cols:
                max_val = max(max_val, df[pct_cols].max().max())

    if is_delta:
        return max(max_val, 5.0)
    return max(max_val, 1.0)


_PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
    "#9c755f", "#bab0ac",
]


def build_color_map(types: list[int]) -> dict:
    return {t: _PALETTE[i % len(_PALETTE)] for i, t in enumerate(types)}


# ---------------------------------------------------------------------------
# Main GIF generation function
# ---------------------------------------------------------------------------

def generate_distribution_gif(
    sim_dir: str | Path,
    num_parts: int = 10,
    exclude_types: tuple | list = (1, 5),
    step: int = 1,
    fps: int = 5,
    output: str | Path | None = None,
    dpi: int = 120,
    mode: str = "percentage",
    show_props: bool = True,
    verbose: bool = True,
) -> Path:
    """
    Generate an animated GIF of the particle-type distribution along X,
    evolving over simulation time.
    """
    sim_dir = Path(sim_dir).resolve()
    if not sim_dir.is_dir():
        raise NotADirectoryError(f"Simulation directory not found: '{sim_dir}'")

    if output is None:
        output = sim_dir / "particle_distribution.gif"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[GIF] Scanning for dump files in '{sim_dir / 'results'}'...")
    all_pairs = discover_dump_files(sim_dir)
    selected_pairs = all_pairs[::step]
    n_total = len(selected_pairs)

    if n_total == 0:
        raise ValueError("No dump files selected. Check --step and the results directory.")

    if verbose:
        print(f"[GIF] Found {len(all_pairs)} dump file(s); using {n_total} (step={step}).")

    # Extract simulation dt
    dt = extract_dt_from_files(sim_dir)
    if dt is None and sim_dir.parent != sim_dir:
        dt = extract_dt_from_files(sim_dir.parent)
    if dt is None:
        warnings.warn(
            "Could not determine timestep 'dt' from log/input files. "
            "Physical time labels will be omitted.",
            UserWarning,
            stacklevel=2,
        )
    elif verbose:
        print(f"[GIF] Detected dt = {dt:.3e} s")

    first_file = selected_pairs[0][1]
    header = get_dump_file_header(sim_dir, dump_file_path=str(first_file))

    all_results: list[dict] = []
    all_frame_numbers: list[int] = []
    all_physical_times: list[float | None] = []
    unique_types: set[int] = set()

    if verbose:
        print("[GIF] Analysing dump frames...")

    for idx, (frame_num, fpath) in enumerate(selected_pairs):
        if verbose and (idx % max(1, n_total // 10) == 0 or idx == n_total - 1):
            print(f"      [{idx + 1}/{n_total}] Frame {frame_num} ({fpath.name})")

        try:
            meta, df = read_single_dump(str(fpath), columns=header)
            res = analyze_particle_distribution_x(
                dump_input=(meta, df),
                num_parts=num_parts,
                exclude_types=list(exclude_types),
                use_box_bounds=True,
                sim_dir=sim_dir,
                verbose=False,
            )
            all_results.append(res)
            all_frame_numbers.append(frame_num)

            phys_t = (frame_num * dt) if dt is not None else None
            all_physical_times.append(phys_t)

            df_summary = res["summary_table"]
            for col in df_summary.columns:
                m = re.match(r"type_(\d+)_pct", col)
                if m:
                    unique_types.add(int(m.group(1)))

        except Exception as exc:
            warnings.warn(f"Skipping frame {frame_num} ({fpath.name}): {exc}", stacklevel=2)

    if not all_results:
        raise RuntimeError("All frames failed to process. Check dump-file format and exclude_types.")

    if verbose:
        print(f"[GIF] Successfully processed {len(all_results)} frame(s).")

    sorted_types = sorted(unique_types)
    color_map = build_color_map(sorted_types)

    ref_res = all_results[0]
    global_y_max = compute_global_y_max(all_results, mode, ref_results=ref_res)
    if mode == "percentage":
        global_y_max = 100.0

    if verbose:
        print("[GIF] Rendering animation frames...")

    pil_frames: list[Image.Image] = []
    for idx, (res, frame_num, phys_t) in enumerate(
        zip(all_results, all_frame_numbers, all_physical_times)
    ):
        img = render_frame(
            results=res,
            frame_number=frame_num,
            physical_time=phys_t,
            dt=dt,
            mode=mode,
            show_props=show_props,
            dpi=dpi,
            global_y_max=global_y_max,
            frame_index=idx,
            total_frames=len(all_results),
            color_map=color_map,
            ref_results=ref_res,
        )
        pil_frames.append(img.convert("RGBA"))

    duration_ms = max(int(1000 / fps), 20)

    if verbose:
        print(f"[GIF] Encoding {len(pil_frames)} frame(s) at {fps} fps -> '{output}'")

    pil_frames[0].save(
        output,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )

    if verbose:
        size_mb = output.stat().st_size / 1_048_576
        print(f"[GIF] Saved '{output}' ({size_mb:.2f} MB)")

    return output.resolve()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an animated GIF of particle distribution along X from dump files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sim_dir",
        type=str,
        required=True,
        help="Path to the simulation directory containing results/ and log/input files.",
    )
    parser.add_argument(
        "--num_parts",
        type=int,
        default=10,
        help="Number of equal X-bins for the distribution.",
    )
    parser.add_argument(
        "--exclude_types",
        type=int,
        nargs="*",
        default=[1, 5],
        help="Particle types to exclude (e.g. floor=1, wall=5).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Use every N-th dump file (1 = all frames).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second in the output GIF.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output GIF path. Defaults to <sim_dir>/particle_distribution.gif.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Figure DPI for each rendered frame.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="percentage",
        choices=[
            "percentage",
            "count",
            "lines",
            "delta_percentage",
            "delta_pct",
            "change",
            "delta_lines",
        ],
        help=(
            "Visualization mode: "
            "'percentage' (stacked bar), 'count' (grouped count bar), 'lines' (profile lines), "
            "'delta_percentage' / 'delta_pct' / 'change' (grouped bar of Δ%% relative to t0), "
            "'delta_lines' (profile lines of Δ%%)."
        ),
    )
    parser.add_argument(
        "--no_props",
        action="store_true",
        help="Suppress the material-properties box on each frame.",
    )

    args = parser.parse_args()

    generate_distribution_gif(
        sim_dir=args.sim_dir,
        num_parts=args.num_parts,
        exclude_types=args.exclude_types,
        step=args.step,
        fps=args.fps,
        output=args.output,
        dpi=args.dpi,
        mode=args.mode,
        show_props=not args.no_props,
        verbose=True,
    )
