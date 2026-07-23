from __future__ import annotations

import re
import time
from pathlib import Path

import matplotlib

# Required for command-line and HPC runs without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


CLASS_ORDER = ["noise", "ambiguous", "likely_xray", "likely_streak"]
CLASS_COLORS = {
    "noise": "0.55",
    "ambiguous": "tab:orange",
    "likely_xray": "tab:blue",
    "likely_streak": "tab:red",
}
COLUMN_LABELS = {
    "sum5x5_bgsub_DN": ("Background-subtracted 5×5 signal (DN)"),
    "minor_axis_extent_phase": ("Minor-axis extent + (y mod 128)/128"),
    "major_axis_extent_phase": ("Major-axis extent + (x mod 128)/128"),
}

def _slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    return text.strip("_").lower()


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "is_sim" not in out.columns:
        out["is_sim"] = False
    elif not pd.api.types.is_bool_dtype(out["is_sim"]):
        out["is_sim"] = (
            out["is_sim"].astype(str).str.strip().str.lower()
            .isin({"true", "1", "yes", "y"})
        )
    else:
        out["is_sim"] = out["is_sim"].fillna(False)

    if "class" not in out.columns:
        out["class"] = "unclassified"
    else:
        out["class"] = out["class"].fillna("unclassified").astype(str)

    if "annular_excess" not in out and {"r3", "r5"}.issubset(out.columns):
        out["annular_excess"] = out["r5"] - out["r3"]

    # Break extent quantization using sub-supercell detector position.
    if (
        "minor_axis_extent_phase" not in out.columns
        and {"minor_axis_extent", "y"}.issubset(out.columns)
    ):
        minor_extent = pd.to_numeric(
            out["minor_axis_extent"],
            errors="coerce",
        )
        y_position = pd.to_numeric(
            out["y"],
            errors="coerce",
        )

        out["minor_axis_extent_phase"] = (
            minor_extent
            + (y_position % 128) / 128.0
        )


    if (
        "major_axis_extent_phase" not in out.columns
        and {"major_axis_extent", "x"}.issubset(out.columns)
    ):
        major_extent = pd.to_numeric(
            out["major_axis_extent"],
            errors="coerce",
        )
        x_position = pd.to_numeric(
            out["x"],
            errors="coerce",
        )

        out["major_axis_extent_phase"] = (
            major_extent
            + (x_position % 128) / 128.0
        )

    return out


def _ordered_classes(df: pd.DataFrame) -> list[str]:
    present = list(pd.unique(df["class"]))
    ordered = [name for name in CLASS_ORDER if name in present]
    ordered.extend(sorted(name for name in present if name not in ordered))
    return ordered


def _class_color(class_name: str) -> str:
    return CLASS_COLORS.get(class_name, "tab:purple")


def _finite_rows(
    df: pd.DataFrame,
    x: str,
    y: str | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
) -> pd.DataFrame:
    columns = [x] if y is None else [x, y]
    out = df.copy()

    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    mask = np.ones(len(out), dtype=bool)
    for column in columns:
        mask &= np.isfinite(out[column].to_numpy(dtype=float))

    if xscale == "log":
        mask &= out[x].to_numpy(dtype=float) > 0
    if y is not None and yscale == "log":
        mask &= out[y].to_numpy(dtype=float) > 0

    return out.loc[mask].copy()


def _sample_real(
    df: pd.DataFrame,
    max_real_per_class: int | None,
    random_seed: int,
) -> tuple[pd.DataFrame, int, int]:
    sim = df.loc[df["is_sim"]]
    real = df.loc[~df["is_sim"]]

    if max_real_per_class is None:
        return df, len(real), len(real)

    pieces = [sim]
    shown_real = 0

    for class_index, (_, group) in enumerate(real.groupby("class", sort=False)):
        if len(group) > max_real_per_class:
            group = group.sample(
                n=max_real_per_class,
                random_state=random_seed + class_index,
            )
        pieces.append(group)
        shown_real += len(group)

    return pd.concat(pieces).sort_index(), shown_real, len(real)


def _add_scatter_legends(ax: plt.Axes, classes: list[str]) -> None:
    class_handles = [
        Patch(facecolor=_class_color(name), edgecolor="none", label=name)
        for name in classes
    ]
    class_legend = ax.legend(
        handles=class_handles,
        title="Classification",
        loc="upper left",
        frameon=True,
    )
    ax.add_artist(class_legend)

    origin_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor="0.35", markeredgecolor="none",
            markersize=5, alpha=0.22, label="Real",
        ),
        Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor="white", markeredgecolor="black",
            markersize=9, alpha=1.0, label="Simulated",
        ),
    ]
    ax.legend(
        handles=origin_handles,
        title="Origin",
        loc="upper right",
        frameon=True,
    )


def save_class_origin_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    output_path: str | Path,
    *,
    title: str | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    max_real_per_class: int | None = 10000,
    random_seed: int = 12345,
    dpi: int = 180,
) -> Path | None:
    """Color encodes class; size and opacity encode real versus simulated."""
    if x not in df.columns or y not in df.columns:
        return None

    work = _finite_rows(
        _prepare_dataframe(df), x=x, y=y, xscale=xscale, yscale=yscale
    )
    if work.empty:
        return None

    plotted, shown_real, total_real = _sample_real(
        work,
        max_real_per_class=max_real_per_class,
        random_seed=random_seed,
    )
    classes = _ordered_classes(plotted)

    fig, ax = plt.subplots(figsize=(8.2, 6.4))

    # Real events first: small and translucent.
    for class_name in classes:
        group = plotted.loc[
            (plotted["class"] == class_name) & (~plotted["is_sim"])
        ]
        if not group.empty:
            ax.scatter(
                group[x], group[y], s=10, alpha=0.13,
                c=_class_color(class_name), linewidths=0,
                rasterized=True,
            )

    # Simulated events last: large, opaque, and outlined.
    for class_name in classes:
        group = plotted.loc[
            (plotted["class"] == class_name) & plotted["is_sim"]
        ]
        if not group.empty:
            ax.scatter(
                group[x], group[y], s=72, alpha=0.95,
                c=_class_color(class_name), edgecolors="black",
                linewidths=0.8, zorder=5,
            )

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(COLUMN_LABELS.get(x, x))
    ax.set_ylabel(COLUMN_LABELS.get(y, y))
    ax.set_title(title or f"{COLUMN_LABELS.get(y, y)} versus {COLUMN_LABELS.get(x, x)}")
    ax.grid(True, alpha=0.22)
    _add_scatter_legends(ax, classes)

    n_sim = int(work["is_sim"].sum())
    note = f"Simulated shown: {n_sim} | Real shown: {shown_real:,}/{total_real:,}"
    ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=8.5)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _common_bins(values: np.ndarray, bins: int, xscale: str) -> np.ndarray:
    values = values[np.isfinite(values)]
    if xscale == "log":
        values = values[values > 0]
    if values.size == 0:
        return np.array([])

    lo = float(values.min())
    hi = float(values.max())
    if lo == hi:
        width = max(abs(lo) * 0.05, 0.5)
        return np.linspace(lo - width, hi + width, bins + 1)
    if xscale == "log":
        return np.geomspace(lo, hi, bins + 1)
    return np.linspace(lo, hi, bins + 1)


def save_class_origin_histogram(
    df: pd.DataFrame,
    column: str,
    output_path: str | Path,
    *,
    title: str | None = None,
    xscale: str = "linear",
    bins: int = 45,
    density: bool = True,
    dpi: int = 180,
) -> Path | None:
    """Normalized class-colored histograms with solid real and dashed sim lines."""
    if column not in df.columns:
        return None

    work = _finite_rows(_prepare_dataframe(df), x=column, xscale=xscale)
    if work.empty:
        return None

    common_bins = _common_bins(
        work[column].to_numpy(dtype=float), bins=bins, xscale=xscale
    )
    if common_bins.size == 0:
        return None

    classes = _ordered_classes(work)
    fig, ax = plt.subplots(figsize=(8.2, 6.0))

    for class_name in classes:
        group = work.loc[work["class"] == class_name]
        real_values = group.loc[~group["is_sim"], column].to_numpy(dtype=float)
        sim_values = group.loc[group["is_sim"], column].to_numpy(dtype=float)

        if real_values.size:
            ax.hist(
                real_values, bins=common_bins, density=density,
                histtype="step", linewidth=1.4, alpha=0.48,
                color=_class_color(class_name), linestyle="-",
            )

        if sim_values.size >= 2:
            ax.hist(
                sim_values, bins=common_bins, density=density,
                histtype="step", linewidth=2.3, alpha=0.98,
                color=_class_color(class_name), linestyle="--",
            )
        elif sim_values.size == 1:
            ax.axvline(
                sim_values[0], linewidth=2.3, alpha=0.98,
                color=_class_color(class_name), linestyle="--",
            )

    ax.set_xscale(xscale)
    ax.set_xlabel(column)
    ax.set_ylabel("Probability density" if density else "Count")
    ax.set_title(title or f"Distribution of {column}")
    ax.grid(True, alpha=0.22)

    class_handles = [
        Patch(facecolor=_class_color(name), edgecolor="none", label=name)
        for name in classes
    ]
    class_legend = ax.legend(
        handles=class_handles,
        title="Classification",
        loc="upper left",
        frameon=True,
    )
    ax.add_artist(class_legend)

    ax.legend(
        handles=[
            Line2D([0], [0], color="0.25", linewidth=1.5,
                   alpha=0.5, linestyle="-", label="Real"),
            Line2D([0], [0], color="0.25", linewidth=2.3,
                   alpha=1.0, linestyle="--", label="Simulated"),
        ],
        title="Origin",
        loc="upper right",
        frameon=True,
    )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_diagnostic_plots(
    pre_df: pd.DataFrame,
    df_streaks: pd.DataFrame | None = None,
    *,
    output_root: str | Path = "cr_event_analysis_plots",
    timestamp: str | None = None,
    max_real_per_class: int | None = 10000,
    random_seed: int = 12345,
    dpi: int = 180,
) -> Path:
    """Create curated scatter plots, histograms, and a plot manifest."""
    timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_directory = Path(output_root) / f"plots_{timestamp}"
    scatter_directory = run_directory / "scatter"
    histogram_directory = run_directory / "histograms"
    scatter_directory.mkdir(parents=True, exist_ok=True)
    histogram_directory.mkdir(parents=True, exist_ok=True)

    pre = _prepare_dataframe(pre_df)
    streaks = None
    if df_streaks is not None and not df_streaks.empty:
        streaks = _prepare_dataframe(df_streaks)

    pre_scatter_specs = [
        ("peak_val", "r3", "log", "linear"),
        ("peak_val", "r5", "log", "linear"),
        ("peak_val", "annular_excess", "log", "linear"),
        ("peak_val", "linearity", "log", "linear"),
        ("peak_val", "anisotropy", "log", "linear"),
        ("peak_val", "aspect_ratio", "log", "linear"),
        ("r3", "r5", "linear", "linear"),
        ("linearity", "anisotropy", "linear", "linear"),
    # Quantization-expanded extent plots
        ("minor_axis_extent_phase", "sum5x5_bgsub_DN", "log", "linear"),
        ("major_axis_extent_phase", "sum5x5_bgsub_DN", "log", "linear"),

    ]
    pre_hist_specs = [
        ("peak_val", "log"),
        ("r3", "linear"),
        ("r5", "linear"),
        ("annular_excess", "linear"),
        ("linearity", "linear"),
        ("anisotropy", "linear"),
        ("aspect_ratio", "linear"),
        ("major_axis_extent", "linear"),
    ]

    streak_scatter_specs = [
        ("blob_e", "n_pix_blob", "log", "linear"),
        ("blob_e", "major_extent_pix", "log", "linear"),
        ("blob_e", "aspect_ratio_blob", "log", "linear"),
        ("blob_e", "gini_blob", "log", "linear"),
        ("major_extent_pix", "minor_extent_pix", "linear", "linear"),
        ("n_pix_blob", "gini_blob", "linear", "linear"),
        ("peak_val", "annular_excess", "log", "linear"),
    ]
    streak_hist_specs = [
        ("blob_e", "log"),
        ("n_pix_blob", "linear"),
        ("major_extent_pix", "linear"),
        ("minor_extent_pix", "linear"),
        ("aspect_ratio_blob", "linear"),
        ("gini_blob", "linear"),
        ("annular_excess", "linear"),
    ]

    manifest: list[dict[str, object]] = []

    def record(dataset: str, kind: str, x: str, y: str | None, path: Path | None):
        manifest.append({
            "dataset": dataset,
            "plot_type": kind,
            "x": x,
            "y": y,
            "status": "saved" if path is not None else "skipped",
            "path": str(path) if path is not None else "",
        })

    for x, y, xscale, yscale in pre_scatter_specs:
        path = save_class_origin_scatter(
            pre, x=x, y=y,
            output_path=scatter_directory / f"pre_{_slug(y)}_vs_{_slug(x)}.png",
            title=f"Preclassification: {y} versus {x}",
            xscale=xscale, yscale=yscale,
            max_real_per_class=max_real_per_class,
            random_seed=random_seed, dpi=dpi,
        )
        record("preclassification", "scatter", x, y, path)

    for column, xscale in pre_hist_specs:
        path = save_class_origin_histogram(
            pre, column=column,
            output_path=histogram_directory / f"pre_hist_{_slug(column)}.png",
            title=f"Preclassification distribution: {column}",
            xscale=xscale, dpi=dpi,
        )
        record("preclassification", "histogram", column, None, path)

    if streaks is not None:
        for x, y, xscale, yscale in streak_scatter_specs:
            path = save_class_origin_scatter(
                streaks, x=x, y=y,
                output_path=scatter_directory / f"streak_{_slug(y)}_vs_{_slug(x)}.png",
                title=f"Streak candidates: {y} versus {x}",
                xscale=xscale, yscale=yscale,
                max_real_per_class=max_real_per_class,
                random_seed=random_seed, dpi=dpi,
            )
            record("streaks", "scatter", x, y, path)

        for column, xscale in streak_hist_specs:
            path = save_class_origin_histogram(
                streaks, column=column,
                output_path=histogram_directory / f"streak_hist_{_slug(column)}.png",
                title=f"Streak-candidate distribution: {column}",
                xscale=xscale, dpi=dpi,
            )
            record("streaks", "histogram", column, None, path)

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(run_directory / "plot_manifest.csv", index=False)

    n_saved = int((manifest_df["status"] == "saved").sum())
    n_skipped = int((manifest_df["status"] == "skipped").sum())
    print(
        f"Saved {n_saved} diagnostic plots to {run_directory} "
        f"({n_skipped} unavailable-column plots skipped)."
    )
    return run_directory
