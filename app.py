"""
Streamlit visualizer — domain clustering strategy (step by step).

Run with:
    streamlit run app.py
"""

from collections import Counter
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Domain Clustering", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #ffd6f5 0%, #e0c3fc 50%, #c9b6ff 100%);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffc9f2 0%, #d8bbff 100%);
    }
    h1, h2, h3 {
        color: #8e2de2 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
# Next to the script in the toolkit repo, under data/ in the standalone public repo.
DOMAINS_CSV = next(
    (p for p in (SCRIPT_DIR / "tf_domains.csv", SCRIPT_DIR / "data" / "tf_domains.csv") if p.exists()),
    SCRIPT_DIR / "tf_domains.csv",
)

ALL_SOURCES = ["merizo", "chainsaw", "unidoc-ndr", "uniprot"]
SOURCE_COLORS = {
    "merizo": "#4C72B0",
    "chainsaw": "#DD8452",
    "unidoc-ndr": "#55A868",
    "uniprot": "#C44E52",
}

# UniProt feature types written by fetch_tf_dataset.py (column `feature_type`, only filled for
# source == "uniprot"). Zinc fingers are numerous (one row per finger): keep them opt-in.
UNIPROT_FEATURE_TYPES = ["Domain", "Zinc finger", "DNA binding"]
DEFAULT_UNIPROT_FEATURE_TYPES = ["Domain"]
# Visual mark of each UniProt feature type: a colour variation of the UniProt red (step 1) and a
# hatch pattern (every step, since the colour is then the cluster's).
UNIPROT_TYPE_STYLE = {
    "Domain": {"color": "#C44E52", "hatch": ""},
    "Zinc finger": {"color": "#D9777B", "hatch": "///"},
    "DNA binding": {"color": "#8E2A2E", "hatch": "xxx"},
}

CLUSTER_COLORS = [
    "#3498DB",
    "#2ECC71",
    "#F4D03F",
    "#E67E22",
    "#9B59B6",
    "#1ABC9C",
    "#CA6F1E",
    "#85929E",
    "#EB984E",
    "#5DADE2",
]


def step_titles(metric_label: str = "IoU") -> list[str]:
    return [
        "Step 1 — All domains",
        f"Step 2 — Clustering (connected components, {metric_label} ≥ threshold)",
        "Step 3 — Representative selection",
        "Step 4 — Final result",
    ]


# ── Data loading ──────────────────────────────────────────────────────────────


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DOMAINS_CSV)
    df = df[df["source"].isin(ALL_SOURCES)].copy()
    for col in ("start", "end", "mean_plddt"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["feature_type"] = df["feature_type"].fillna("")
    return df


def filter_uniprot_features(df: pd.DataFrame, feature_types: list[str] | tuple[str, ...]):
    """Keep every TED row, and only the UniProt rows whose feature_type is selected."""
    return df[(df["source"] != "uniprot") | df["feature_type"].isin(feature_types)]


# ── Algorithm ─────────────────────────────────────────────────────────────────

METRICS = {
    "IoU (Jaccard)": "iou",
    "Overlap (Szymkiewicz–Simpson)": "overlap",
    "Dice": "dice",
    "Coverage": "coverage",
}


# One colour per metric (colour-blind safe), used by the global threshold scan.
METRIC_COLORS = dict(zip(METRICS, ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]))


def _inter(s1: int, e1: int, s2: int, e2: int) -> int:
    return max(0, min(e1, e2) - max(s1, s2) + 1)


def similarity(s1: int, e1: int, s2: int, e2: int, metric: str = "iou") -> float:
    inter = _inter(s1, e1, s2, e2)
    if inter == 0:
        return 0.0
    len1 = e1 - s1 + 1
    len2 = e2 - s2 + 1
    if metric == "iou":
        denom = len1 + len2 - inter
    elif metric == "overlap":
        denom = min(len1, len2)
    elif metric == "dice":
        denom = (len1 + len2) / 2
    elif metric == "coverage":
        denom = max(len1, len2)
    else:
        denom = len1 + len2 - inter
    return inter / denom if denom > 0 else 0.0


def find_clusters(doms: list[dict], threshold: float, metric: str = "iou") -> list[list[int]]:
    n = len(doms)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if (
                similarity(
                    doms[i]["start"], doms[i]["end"], doms[j]["start"], doms[j]["end"], metric
                )
                >= threshold
            ):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def multi_domain_clusters_by_source(
    clusters: list[list[int]], doms: list[dict], sources: list[str]
) -> dict[str, int]:
    """Per source: number of clusters holding more than one domain of that source.

    Such a cluster means two domains of the same source were linked indirectly (through a
    domain of another source, or by overlapping each other).
    """
    counts = dict.fromkeys(sources, 0)
    for cluster in clusters:
        for source, n in Counter(doms[i]["source"] for i in cluster).items():
            if n > 1:
                counts[source] += 1
    return counts


SCAN_THRESHOLDS = [round(0.5 + 0.05 * k, 2) for k in range(11)]  # 0.50 → 1.00, pas de 0.05


@st.cache_data(show_spinner="Computing over all proteins…")
def total_domains_by_threshold(
    _df: pd.DataFrame, metric: str, include_unidoc: bool, feature_types: tuple[str, ...]
) -> tuple[list[int], int]:
    """Total number of final domains (= clusters) over all proteins, for each threshold.

    Same rule as find_clusters (connected components, similarity ≥ threshold), but the
    similarities are computed once per protein and reused for every threshold.
    Returns (one total per threshold of SCAN_THRESHOLDS, number of raw domains before clustering).
    `_df`: the `_` prefix excludes the DataFrame from the cache key (hence `feature_types`,
    which is part of it and filters the UniProt features taken into account).
    """
    _df = filter_uniprot_features(_df, feature_types)
    sources = [s for s in ALL_SOURCES if s != "unidoc-ndr" or include_unidoc]
    totals = [0] * len(SCAN_THRESHOLDS)
    n_raw = 0

    for _, g in _df[_df["source"].isin(sources)].groupby("uniprot_id"):
        starts, ends = g["start"].tolist(), g["end"].tolist()
        n = len(g)
        n_raw += n
        edges = []  # (i, j, similarity) for the overlapping pairs only
        for i in range(n):
            for j in range(i + 1, n):
                sim = similarity(starts[i], ends[i], starts[j], ends[j], metric)
                if sim > 0:
                    edges.append((i, j, sim))

        for k, threshold in enumerate(SCAN_THRESHOLDS):
            parent = list(range(n))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            n_clusters = n
            for i, j, sim in edges:
                if sim >= threshold:
                    pi, pj = find(i), find(j)
                    if pi != pj:
                        parent[pi] = pj
                        n_clusters -= 1
            totals[k] += n_clusters
    return totals, n_raw


@st.cache_data(show_spinner="Looking for multi-domain clusters…")
def multi_domain_clusters_by_threshold(
    _df: pd.DataFrame, metric: str, include_unidoc: bool, feature_types: tuple[str, ...]
) -> dict[str, list[int]]:
    """Same count as multi_domain_clusters_by_source, over all proteins, for each threshold.

    Returns {source: [number of clusters holding >= 2 domains of that source, per threshold of
    SCAN_THRESHOLDS]}. As in total_domains_by_threshold, the similarities are computed once per
    protein. `_df` is excluded from the cache key, hence the other parameters.
    """
    _df = filter_uniprot_features(_df, feature_types)
    sources = [s for s in ALL_SOURCES if s != "unidoc-ndr" or include_unidoc]
    counts = {src: [0] * len(SCAN_THRESHOLDS) for src in sources}

    for _, g in _df[_df["source"].isin(sources)].groupby("uniprot_id"):
        starts, ends, srcs = g["start"].tolist(), g["end"].tolist(), g["source"].tolist()
        n = len(g)
        edges = []  # (i, j, similarity) for the overlapping pairs only
        for i in range(n):
            for j in range(i + 1, n):
                sim = similarity(starts[i], ends[i], starts[j], ends[j], metric)
                if sim > 0:
                    edges.append((i, j, sim))

        for k, threshold in enumerate(SCAN_THRESHOLDS):
            parent = list(range(n))

            def find(x, parent=parent):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for i, j, sim in edges:
                if sim >= threshold:
                    pi, pj = find(i), find(j)
                    if pi != pj:
                        parent[pi] = pj
            per_cluster = Counter((find(i), srcs[i]) for i in range(n))
            for (_, src), n_dom in per_cluster.items():
                if n_dom > 1:
                    counts[src][k] += 1
    return counts


def _uniprot_span(cluster_doms: list[dict]) -> tuple[int, int]:
    """Union (min start, max end) of every UniProt domain of the cluster."""
    us = [d for d in cluster_doms if d["source"] == "uniprot"]
    return min(d["start"] for d in us), max(d["end"] for d in us)


def select_rep(cluster_doms: list[dict]) -> dict:
    if len(cluster_doms) == 1:
        d = cluster_doms[0]
        return {k: d.get(k, "") for k in ("start", "end", "source", "mean_plddt", "feature_type")}

    sources = {d["source"] for d in cluster_doms}
    has_uniprot = "uniprot" in sources

    # TED method priority: merizo > chainsaw > unidoc-ndr.
    for method in ("merizo", "chainsaw", "unidoc-ndr"):
        if method not in sources:
            continue
        t = next(d for d in cluster_doms if d["source"] == method)
        if not has_uniprot:
            return {
                "start": t["start"],
                "end": t["end"],
                "source": method,
                "mean_plddt": t["mean_plddt"],
            }
        u_start, u_end = _uniprot_span(cluster_doms)
        return {
            "start": min(t["start"], u_start),
            "end": max(t["end"], u_end),
            "source": f"{method}+uniprot",
            "mean_plddt": None,
        }

    # UniProt only: several features can fall in the same cluster (e.g. adjacent zinc fingers).
    u_start, u_end = _uniprot_span(cluster_doms)
    us = [d for d in cluster_doms if d["source"] == "uniprot"]
    types = {d.get("feature_type", "") for d in us}
    return {
        "start": u_start,
        "end": u_end,
        "source": "uniprot",
        "mean_plddt": us[0]["mean_plddt"] if len(us) == 1 else None,
        "feature_type": types.pop() if len(types) == 1 else "",
    }


def rule_label(cluster_doms: list[dict]) -> str:
    sources = sorted({d["source"] for d in cluster_doms})
    if len(cluster_doms) == 1:
        return f"singleton → {sources[0]}"
    label = " + ".join(sources)
    src = select_rep(cluster_doms)["source"]
    return f"{label} → {src}"


# ── Drawing ───────────────────────────────────────────────────────────────────

BAR_H = 0.42


def _bar(
    ax, start, end, y, color, alpha=0.9, edgecolor="white", lw=1.0, linestyle="-", hatch="", h=BAR_H
):
    width = max(int(end) - int(start), 1)
    rect = plt.Rectangle(
        (int(start), y - h / 2),
        width,
        h,
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=lw,
        alpha=alpha,
        zorder=2,
        linestyle=linestyle,
        hatch=hatch or None,
    )
    ax.add_patch(rect)


def _hatch(domain: dict) -> str:
    """Hatch pattern of a UniProt domain according to its feature type ("" otherwise)."""
    if domain["source"] != "uniprot":
        return ""
    return UNIPROT_TYPE_STYLE.get(domain.get("feature_type", ""), {}).get("hatch", "")


def _label(ax, start, end, y, text, fontsize=7):
    ax.text(
        (int(start) + int(end)) / 2,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="white",
        fontweight="bold",
        zorder=3,
    )


LANE_PITCH = 0.5  # vertical distance between two lanes of the same source
LANE_BAR_H = 0.4  # bar height when a source needs several lanes


def assign_lanes(items: list[dict], sources) -> list[int]:
    """Lane index of each item, so that overlapping items of a same source never share a lane.

    Greedy interval partitioning, per source. Items whose source is not a plain source of
    `sources` (e.g. the "merizo+uniprot" unions) stay on lane 0.
    """
    lanes = [0] * len(items)
    for src in sources:
        lane_ends: list[float] = []
        members = sorted(
            (i for i, it in enumerate(items) if it["source"] == src), key=lambda i: items[i]["start"]
        )
        for i in members:
            for k, end in enumerate(lane_ends):
                if end < items[i]["start"]:
                    lane_ends[k] = items[i]["end"]
                    lanes[i] = k
                    break
            else:
                lanes[i] = len(lane_ends)
                lane_ends.append(items[i]["end"])
    return lanes


UNION_ROW = "merged"  # label of the row holding the TED+UniProt union representatives


def compute_layout(domains, reps, source_y, with_unions=False):
    """Vertical layout: one block per source, with one lane per level of overlap.

    With `with_unions`, an extra top block holds the union representatives ("merizo+uniprot"...)
    instead of drawing them between the rows of their sources.

    Returns (y of each domain, y of each representative, centre y of each block, y of every
    lane, number of lanes of the busiest block).
    """
    order = sorted(source_y, key=source_y.get)  # bottom to top
    dom_lane = assign_lanes(domains, order)
    rep_lane = assign_lanes(reps, order)
    union_idx = [i for i, r in enumerate(reps) if "+" in r["source"]] if with_unions else []
    union_lane = dict(
        zip(
            union_idx,
            assign_lanes(
                [{"source": UNION_ROW, "start": reps[i]["start"], "end": reps[i]["end"]} for i in union_idx],
                [UNION_ROW],
            ),
        )
    )
    n_lanes = {src: 1 for src in order}
    for items, lanes in ((domains, dom_lane), (reps, rep_lane)):
        for it, lane in zip(items, lanes):
            if it["source"] in n_lanes:
                n_lanes[it["source"]] = max(n_lanes[it["source"]], lane + 1)
    if union_idx:
        n_lanes[UNION_ROW] = 1 + max(union_lane.values())
        order = [*order, UNION_ROW]

    base, center, lane_ys, cur = {}, {}, [], 0.0
    for src in order:
        n = n_lanes[src]
        pitch = LANE_PITCH if n > 1 else 0.0
        base[src] = cur
        center[src] = cur + (n - 1) * pitch / 2
        lane_ys += [cur + k * pitch for k in range(n)]
        cur += (n - 1) * pitch + 1

    def y_of(src, lane):
        return base[src] + lane * (LANE_PITCH if n_lanes[src] > 1 else 0.0)

    y_dom = [y_of(d["source"], lane) for d, lane in zip(domains, dom_lane)]
    y_rep = [
        y_of(UNION_ROW, union_lane[i]) if i in union_lane else y_of(r["source"], rep_lane[i])
        if r["source"] in base
        else 1
        for i, r in enumerate(reps)
    ]
    return y_dom, y_rep, center, lane_ys, max(n_lanes.values())


def draw_step(domains, clusters, reps, step, prot_len, threshold, source_y, metric="iou"):
    y_dom, y_rep, center, lane_ys, max_lanes = compute_layout(
        domains, reps, source_y, with_unions=step >= 2
    )
    bar_h = LANE_BAR_H if max_lanes > 1 else BAR_H

    def bar(*args, **kwargs):
        _bar(*args, h=bar_h, **kwargs)

    total_height = max(lane_ys) + 1 if lane_ys else 1
    fig, ax = plt.subplots(figsize=(14, 2.4 + total_height * 0.7))

    cluster_of = {idx: ci for ci, cluster in enumerate(clusters) for idx in cluster}

    for y in lane_ys:
        ax.plot([1, prot_len], [y, y], color="#DDDDDD", linewidth=1, zorder=0)

    if step == 0:
        for i, d in enumerate(domains):
            y = y_dom[i]
            style = UNIPROT_TYPE_STYLE.get(d.get("feature_type", "")) if d["source"] == "uniprot" else None
            bar(
                ax,
                d["start"],
                d["end"],
                y,
                style["color"] if style else SOURCE_COLORS[d["source"]],
                hatch=_hatch(d),
            )
            _label(ax, d["start"], d["end"], y, f"{int(d['start'])}–{int(d['end'])}")

    elif step == 1:
        for i, d in enumerate(domains):
            y = y_dom[i]
            color = CLUSTER_COLORS[cluster_of[i] % len(CLUSTER_COLORS)]
            bar(ax, d["start"], d["end"], y, color, edgecolor="#333333", lw=1.2, hatch=_hatch(d))
            _label(ax, d["start"], d["end"], y, f"{int(d['start'])}–{int(d['end'])}")

        for cluster in clusters:
            for a in range(len(cluster)):
                for b in range(a + 1, len(cluster)):
                    d1, d2 = domains[cluster[a]], domains[cluster[b]]
                    y1, y2 = y_dom[cluster[a]], y_dom[cluster[b]]
                    x1 = (d1["start"] + d1["end"]) / 2
                    x2 = (d2["start"] + d2["end"]) / 2
                    val = similarity(d1["start"], d1["end"], d2["start"], d2["end"], metric)
                    ax.plot(
                        [x1, x2], [y1, y2], color="#555555", linewidth=1.2, linestyle=":", zorder=1
                    )
                    ax.text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color="#222222",
                        zorder=4,
                        bbox=dict(
                            boxstyle="round,pad=0.15",
                            facecolor="white",
                            edgecolor="#999999",
                            alpha=0.85,
                        ),
                    )

    elif step == 2:
        contributing: set[tuple[int, str]] = set()
        for ci, rep in enumerate(reps):
            for src in rep["source"].split("+"):
                contributing.add((ci, src))

        for i, d in enumerate(domains):
            ci = cluster_of[i]
            is_rep = (ci, d["source"]) in contributing
            y = y_dom[i]
            color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
            bar(
                ax,
                d["start"],
                d["end"],
                y,
                color,
                alpha=0.9 if is_rep else 0.18,
                edgecolor="black" if is_rep else "white",
                lw=2.0 if is_rep else 0.5,
                hatch=_hatch(d),
            )
            if is_rep:
                _label(ax, d["start"], d["end"], y, f"{int(d['start'])}–{int(d['end'])}")

        for ci, rep in enumerate(reps):
            if "+" not in rep["source"]:
                continue
            y_union = y_rep[ci]
            color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
            bar(
                ax,
                rep["start"],
                rep["end"],
                y_union,
                color,
                alpha=0.75,
                edgecolor="black",
                lw=2.0,
                linestyle="--",
            )
            _label(
                ax, rep["start"], rep["end"], y_union, f"∪ {int(rep['start'])}–{int(rep['end'])}"
            )

    elif step == 3:
        for ci, rep in enumerate(reps):
            color = CLUSTER_COLORS[ci % len(CLUSTER_COLORS)]
            y = y_rep[ci]
            bar(
                ax,
                rep["start"],
                rep["end"],
                y,
                color,
                edgecolor="white",
                lw=1.5,
                hatch=_hatch(rep),
            )
            length = int(rep["end"]) - int(rep["start"]) + 1
            plddt = f" | pLDDT {rep['mean_plddt']:.0f}" if rep.get("mean_plddt") else ""
            _label(
                ax,
                rep["start"],
                rep["end"],
                y,
                f"{int(rep['start'])}–{int(rep['end'])} ({length} aa){plddt}",
            )

    ax.set_xlim(0, prot_len + 20)
    ax.set_ylim(-0.65, max(lane_ys) + 0.85)
    ax.set_yticks(list(center.values()))
    ax.set_yticklabels(list(center.keys()), fontsize=10)
    ax.set_xlabel("Position (residue)", fontsize=10)
    metric_name = {v: k for k, v in METRICS.items()}.get(metric, "IoU")
    ax.set_title(step_titles(metric_name)[step], fontsize=12, fontweight="bold", pad=8)

    present_types = [
        t
        for t in UNIPROT_FEATURE_TYPES
        if any(d["source"] == "uniprot" and d.get("feature_type") == t for d in domains)
    ]
    if step == 0:
        legend_items = []
        for s in source_y:
            if s == "uniprot" and present_types:
                legend_items += [
                    mpatches.Patch(
                        facecolor=UNIPROT_TYPE_STYLE[t]["color"],
                        hatch=UNIPROT_TYPE_STYLE[t]["hatch"] or None,
                        edgecolor="white",
                        label=f"uniprot · {t}",
                    )
                    for t in present_types
                ]
            else:
                legend_items.append(mpatches.Patch(color=SOURCE_COLORS[s], label=s))
    else:
        legend_items = [
            mpatches.Patch(color=CLUSTER_COLORS[ci % len(CLUSTER_COLORS)], label=f"C{ci + 1}")
            for ci in range(len(clusters))
        ]
        if len(present_types) > 1 or present_types not in ([], ["Domain"]):
            legend_items += [
                mpatches.Patch(
                    facecolor="#DDDDDD",
                    hatch=UNIPROT_TYPE_STYLE[t]["hatch"] or None,
                    edgecolor="#333333",
                    label=f"uniprot · {t}",
                )
                for t in present_types
            ]
    ax.legend(
        handles=legend_items,
        loc="upper right",
        fontsize=8,
        framealpha=0.85,
        ncol=2 if len(legend_items) > 6 else 1,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ── Metric illustrations ──────────────────────────────────────────────────────

# Two example segments, deliberately of different lengths so that the metrics differ.
EXAMPLE_A = (1, 40)
EXAMPLE_B = (21, 100)

METRIC_INFO = {
    "iou": (
        "IoU (Jaccard)",
        "|A∩B| / |A∪B|",
        (
            "Denominator = union of the two segments. The strictest metric: two segments of "
            "very different sizes are never close."
        ),
    ),
    "overlap": (
        "Overlap (Szymkiewicz–Simpson)",
        "|A∩B| / min(|A|, |B|)",
        (
            "Denominator = the smaller segment. Equals 1 as soon as one segment is entirely "
            "contained in the other: very permissive, and connected components can chain many "
            "segments together (e.g. zinc fingers inside a DNA-binding region)."
        ),
    ),
    "dice": (
        "Dice",
        "|A∩B| / ((|A| + |B|) / 2)",
        (
            "Denominator = mean of the two lengths (equivalent to 2·|A∩B| / (|A| + |B|)). "
            "Between IoU and Overlap."
        ),
    ),
    "coverage": (
        "Coverage",
        "|A∩B| / max(|A|, |B|)",
        (
            "Denominator = the larger segment: the share of the larger segment covered by "
            "the intersection. Stricter than Dice when the sizes differ."
        ),
    ),
}


DENOMINATOR_LABEL = {
    "iou": "|A∪B|",
    "overlap": "min(|A|, |B|)",
    "dice": "(|A| + |B|) / 2",
    "coverage": "max(|A|, |B|)",
}


def metric_denominator(len_a: int, len_b: int, inter: int, metric: str) -> float:
    return {
        "iou": len_a + len_b - inter,
        "overlap": min(len_a, len_b),
        "dice": (len_a + len_b) / 2,
        "coverage": max(len_a, len_b),
    }[metric]


def draw_metric_illustration(metric: str):
    """Worked example of one similarity metric on EXAMPLE_A / EXAMPLE_B."""
    name, formula, _ = METRIC_INFO[metric]
    (s1, e1), (s2, e2) = EXAMPLE_A, EXAMPLE_B
    len_a, len_b = e1 - s1 + 1, e2 - s2 + 1
    inter = _inter(s1, e1, s2, e2)
    denom = metric_denominator(len_a, len_b, inter, metric)
    value = similarity(s1, e1, s2, e2, metric)
    i_start = max(s1, s2)

    fig, ax = plt.subplots(figsize=(10, 3.4))
    rows = [  # (y, label, start, length, color)
        (3, f"A  ({len_a} aa)", s1, len_a, "#4C72B0"),
        (2, f"B  ({len_b} aa)", s2, len_b, "#DD8452"),
        (1, f"Numerator  |A∩B| = {inter}", i_start, inter, "#8e2de2"),
        (0, f"Denominator  {DENOMINATOR_LABEL[metric]} = {denom:g}", s1, denom, "#555555"),
    ]
    for y, label, start, length, color in rows:
        _bar(ax, start, start + length - 1, y, color, edgecolor="white", lw=1.0)
        ax.text(max(s1, e2) + 4, y, label, va="center", fontsize=9)
    # Dashed guides showing the intersection on A and B.
    ax.axvspan(i_start, i_start + inter - 1, color="#8e2de2", alpha=0.12, zorder=0)

    ax.set_xlim(0, e2 + 55)
    ax.set_ylim(-0.7, 3.7)
    ax.set_yticks([])
    ax.set_xlabel("Position (residue)", fontsize=10)
    ax.set_title(
        f"{name} :  {formula} = {inter} / {denom:g} = {value:.2f}",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    return fig


# ── Step text explanations ────────────────────────────────────────────────────


def step_text(
    domains, clusters, reps, step, protein_id, threshold, active_sources, metric_label="IoU"
):
    source_counts = {s: sum(1 for d in domains if d["source"] == s) for s in active_sources}

    if step == 0:
        lines = [f"**{len(domains)} domains** loaded for `{protein_id}`:"]
        for s in active_sources:
            if source_counts[s]:
                detail = ""
                if s == "uniprot":
                    by_type = Counter(d.get("feature_type", "") for d in domains if d["source"] == s)
                    detail = " (" + ", ".join(f"{t}: {n}" for t, n in by_type.items()) + ")"
                lines.append(f"- **{s}**: {source_counts[s]} domain(s){detail}")
        return "\n".join(lines)

    if step == 1:
        singletons = sum(1 for c in clusters if len(c) == 1)
        multi = len(clusters) - singletons
        lines = [
            f"**{len(clusters)} cluster(s)** formed with {metric_label} ≥ {threshold:.2f}:",
            f"- {singletons} singleton(s)",
            f"- {multi} multi-domain cluster(s)",
            "",
        ]
        for ci, cluster in enumerate(clusters):
            srcs = ", ".join(domains[i]["source"] for i in cluster)
            lines.append(f"- **C{ci + 1}**: {srcs}")
        return "\n".join(lines)

    if step == 2:
        lines = ["**Rule applied per cluster:**", ""]
        for ci, cluster in enumerate(clusters):
            cluster_doms = [domains[i] for i in cluster]
            rep = reps[ci]
            lines.append(
                f"- **C{ci + 1}** [{rule_label(cluster_doms)}]"
                f" → [{int(rep['start'])}–{int(rep['end'])}]"
            )
        return "\n".join(lines)

    # step == 3
    lines = [f"**{len(reps)} final domain(s)** for `{protein_id}`:", ""]
    for rep in reps:
        length = int(rep["end"]) - int(rep["start"]) + 1
        plddt = f" | pLDDT {rep['mean_plddt']:.1f}" if rep.get("mean_plddt") else ""
        lines.append(
            f"- [{int(rep['start'])}–{int(rep['end'])}] ({length} aa) — *{rep['source']}*{plddt}"
        )
    return "\n".join(lines)


# ── App ───────────────────────────────────────────────────────────────────────

df_all = load_data()
proteins = sorted(df_all["uniprot_id"].unique())

st.title("Domain clustering visualizer")

# ── Metric illustrations ─────────────────────────────────────────────────────

st.header("Understanding the similarity metrics")
st.caption(
    f"Example: A = {EXAMPLE_A[0]}–{EXAMPLE_A[1]} and B = {EXAMPLE_B[0]}–{EXAMPLE_B[1]}. "
    "Two segments are in the same cluster if their similarity ≥ threshold "
    "(then connected components: A~B and B~C ⇒ A, B, C together)."
)
for tab, (metric_key, (metric_name, _, description)) in zip(
    st.tabs([info[0] for info in METRIC_INFO.values()]), METRIC_INFO.items()
):
    with tab:
        fig_metric = draw_metric_illustration(metric_key)
        st.pyplot(fig_metric)
        plt.close(fig_metric)
        st.markdown(description)

st.divider()

# Sidebar = data filters that apply to EVERY plot on the page. Metric / threshold are not
# here: each section has its own controls (see below).
with st.sidebar:
    st.header("Data filters")
    st.caption("Apply to all the plots of the page.")
    include_unidoc = st.toggle("Include unidoc-ndr", value=False)
    uniprot_feature_types = st.multiselect(
        "UniProt features",
        UNIPROT_FEATURE_TYPES,
        default=DEFAULT_UNIPROT_FEATURE_TYPES,
        help="UniProt feature types taken into account. \"Zinc finger\" adds one domain per "
        "zinc finger, which greatly inflates the number of domains.",
    )

df = filter_uniprot_features(df_all, uniprot_feature_types)

# ── Section 1 controls: one protein, step by step ────────────────────────────

st.header("Step-by-step view (one protein)")
col_prot, col_metric, col_thr = st.columns([2, 2, 3])
with col_prot:
    protein_id = st.selectbox("Protein", proteins)
with col_metric:
    metric_label = st.selectbox("Metric", list(METRICS.keys()))
    metric = METRICS[metric_label]
with col_thr:
    threshold = st.slider("Threshold", min_value=0.3, max_value=0.9, value=0.6, step=0.05)

# Active sources & dynamic Y positions
active_sources = [s for s in ALL_SOURCES if s != "unidoc-ndr" or include_unidoc]
source_y = {src: i for i, src in enumerate(reversed(active_sources))}

# Reset step when key inputs change
for key, val in [
    ("_last_prot", protein_id),
    ("_last_thr", threshold),
    ("_last_unidoc", include_unidoc),
    ("_last_metric", metric),
    ("_last_ftypes", tuple(uniprot_feature_types)),
]:
    if st.session_state.get(key) != val:
        st.session_state["step"] = 0
        st.session_state[key] = val
        if key == "_last_ftypes":  # manual merges were made on another set of domains
            st.session_state.pop("_domains_override", None)
            st.session_state.pop("_domains_override_prot", None)

if "step" not in st.session_state:
    st.session_state.step = 0

# ── Get working domains (raw or with user merges) ────────────────────────────

prot_df = df[(df["uniprot_id"] == protein_id) & (df["source"].isin(active_sources))]
raw_domains = prot_df[["start", "end", "source", "mean_plddt", "feature_type"]].to_dict("records")

override = st.session_state.get("_domains_override")
override_prot = st.session_state.get("_domains_override_prot")
if override is not None and override_prot == protein_id:
    domains = [d for d in override if d["source"] in active_sources]
else:
    domains = raw_domains

prot_len = int(prot_df["end"].max()) if len(prot_df) else 1

# ── Merge UI ─────────────────────────────────────────────────────────────────

with st.expander("Manual merge"):
    domain_labels = [
        f"[{i}] {d['source']}{' · ' + d['feature_type'] if d.get('feature_type') else ''}"
        f"  {d['start']}–{d['end']}"
        for i, d in enumerate(domains)
    ]
    selected = st.multiselect(
        "Select domains to merge",
        options=list(range(len(domains))),
        format_func=lambda i: domain_labels[i],
    )

    col_m, col_r = st.columns(2)
    with col_m:
        do_merge = st.button("Merge", type="primary", disabled=len(selected) < 2)
    with col_r:
        do_reset = st.button("Reset")

    if do_merge and len(selected) >= 2:
        sources_sel = {domains[i]["source"] for i in selected}
        if len(sources_sel) != 1:
            st.error("Selected domains must come from the same source.")
        else:
            source = next(iter(sources_sel))
            new_start = min(domains[i]["start"] for i in selected)
            new_end = max(domains[i]["end"] for i in selected)
            types_sel = {domains[i].get("feature_type", "") for i in selected}
            merged = {
                "start": new_start,
                "end": new_end,
                "source": source,
                "mean_plddt": None,
                "feature_type": types_sel.pop() if len(types_sel) == 1 else "",
            }
            remaining = [d for i, d in enumerate(domains) if i not in set(selected)]
            remaining.append(merged)
            st.session_state["_domains_override"] = remaining
            st.session_state["_domains_override_prot"] = protein_id
            st.rerun()

    if do_reset:
        st.session_state.pop("_domains_override", None)
        st.session_state.pop("_domains_override_prot", None)
        st.rerun()

# ── Compute ──────────────────────────────────────────────────────────────────

clusters = find_clusters(domains, threshold, metric)
reps = [select_rep([domains[i] for i in c]) for c in clusters]

# ── Navigation ───────────────────────────────────────────────────────────────

col_prev, col_prog, col_next = st.columns([1, 8, 1])
with col_prev:
    if st.button("← Prev.", disabled=st.session_state.step == 0):
        st.session_state.step -= 1
        st.rerun()
with col_next:
    if st.button("Next →", disabled=st.session_state.step == 3):
        st.session_state.step += 1
        st.rerun()

step = st.session_state.step
st.progress((step + 1) / 4, text=f"Step {step + 1} / 4")

# ── Plot ─────────────────────────────────────────────────────────────────────

fig = draw_step(domains, clusters, reps, step, prot_len, threshold, source_y, metric)
st.pyplot(fig)
plt.close(fig)

# ── Explanation ──────────────────────────────────────────────────────────────

st.markdown(
    step_text(domains, clusters, reps, step, protein_id, threshold, active_sources, metric_label)
)

multi_current = multi_domain_clusters_by_source(clusters, domains, active_sources)
if any(multi_current.values()):
    st.warning(
        "Clusters containing several domains of the same source for this protein: "
        + ", ".join(f"**{src}**: {n}" for src, n in multi_current.items() if n)
    )
else:
    st.success("No cluster contains several domains of the same source (this protein).")

# ── Global scan: total number of domains vs threshold ────────────────────────

st.divider()
st.header("Total number of domains by threshold (all proteins)")
st.caption("Compares the four metrics at every threshold: independent of the controls above.")

scan_by_metric: dict[str, list[int]] = {}
n_raw_domains = 0
for label, key in METRICS.items():
    scan_by_metric[label], n_raw_domains = total_domains_by_threshold(
        df_all, key, include_unidoc, tuple(uniprot_feature_types)
    )

threshold_labels = [f"{t:.2f}" for t in SCAN_THRESHOLDS]
fig_scan = go.Figure()
for label, totals in scan_by_metric.items():
    fig_scan.add_bar(
        x=threshold_labels,
        y=totals,
        name=label,
        marker_color=METRIC_COLORS[label],
        hovertemplate="%{x}: %{y:,} domains<extra>" + label + "</extra>",
    )
fig_scan.add_hline(
    y=n_raw_domains,
    line_dash="dash",
    line_color="#555555",
    annotation_text=f"raw domains: {n_raw_domains:,}".replace(",", " "),
    annotation_position="top left",
)
fig_scan.update_layout(
    barmode="group",
    height=420,
    xaxis_title="Threshold",
    yaxis_title="Final domains (total)",
    yaxis_range=[0, n_raw_domains * 1.12],
    legend_title_text="Metric (click = hide / show)",
    margin={"l": 10, "r": 10, "t": 30, "b": 10},
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.6)",
)
st.plotly_chart(fig_scan, width="stretch")

with st.expander("Exact values"):
    st.dataframe(
        pd.DataFrame(scan_by_metric, index=[f"{t:.2f}" for t in SCAN_THRESHOLDS]).rename_axis(
            "Threshold"
        )
    )

st.caption(
    f"{df['uniprot_id'].nunique():,} proteins".replace(",", " ")
    + f" · sources: {', '.join(active_sources)} · dashed line = number of domains before clustering."
    " Click the legend to hide or show a metric."
    " The higher the threshold, the less we merge: the total climbs back towards the raw domain count."
)

# ── Global: clusters with several domains of the same source ─────────────────

st.divider()
st.header("Clusters with several domains of the same source (all proteins)")
multi_metric_label = st.selectbox("Metric", list(METRICS.keys()), key="multi_metric")
multi_counts = multi_domain_clusters_by_threshold(
    df_all, METRICS[multi_metric_label], include_unidoc, tuple(uniprot_feature_types)
)

fig_multi = go.Figure()
for src, counts in multi_counts.items():
    fig_multi.add_bar(
        x=threshold_labels,
        y=counts,
        name=src,
        marker_color=SOURCE_COLORS[src],
        hovertemplate="%{x}: %{y:,} clusters<extra>" + src + "</extra>",
    )
fig_multi.update_layout(
    barmode="group",
    height=420,
    xaxis_title="Threshold",
    yaxis_title="Clusters with ≥ 2 domains of the same source",
    legend_title_text="Source (click = hide / show)",
    margin={"l": 10, "r": 10, "t": 30, "b": 10},
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.6)",
)
st.plotly_chart(fig_multi, width="stretch")

with st.expander("Exact values (clusters with several domains of the same source)"):
    st.dataframe(pd.DataFrame(multi_counts, index=threshold_labels).rename_axis("Threshold"))

st.caption(
    f"{multi_metric_label}. A cluster counts for a source if it contains at least 2 of its "
    "domains: they were linked indirectly (through a domain of another source, or by "
    "overlapping). A higher threshold reduces this number."
)
