"""
PP-attachment: full model analysis, steps 1-5 (steps 3b and 5 need the human study).

Run after eval_models.py and, for the human steps, human_prep.py:
  python scripts/model_analysis.py                 # defaults below
  python scripts/model_analysis.py --no-humans     # steps 1-4 only
Defaults: --models results/nli_scores_<PROMPT_FORMAT>.csv, --key/--v1/--v2 from data/human_results/,
          --out results/analysis/

Sign convention everywhere: bias = P(E | Px, H_low) - P(E | Px, H_high)
  > 0  model prefers the Low reading (recency),  < 0  prefers High.
Humans: lean from -2 (High) to +2 (Low), same sign.

Human scoring and exclusions come from human_analysis.py. Pairwise control accuracy uses a strict
P(E|ent) > P(E|non) (no tie rule), so it can differ slightly from summary_<PROMPT_FORMAT>.csv.
Human-ambiguous items: MAIN = 95% CI of the item's mean lean includes 0 ('ns');
ROBUSTNESS = |mean lean| < --balanced. Wilcoxon (vs 0) and Mann-Whitney (subset vs rest) p-values.

Outputs (in --out):
  step1_controls.csv                 control accuracy per model x category (+ flags < 0.90)
  step2_contradiction.csv            P(C) on controls / ambiguous pairs, 'neither entailed' counts, how the loser is rejected
  (figures are written as .png and .pdf, sized for a two-column paper like the human figures)
  step3_bias.csv, fig_step3_bias     bias box plots (all items) + winner<0.5 / gap flags
  step3_bias_ns.csv, fig_step3_bias_ns               same, only items without a significant human lean (MAIN)
  step3_bias_balanced.csv, fig_step3_bias_balanced   same, only items humans find balanced (robustness)
  step3_all_vs_ns.csv, step3_all_vs_balanced.csv     all items vs subset vs rest, with p-values
  step4_jsd.csv, fig_step4_jsd      JS distance box plots
  step5_human_human                 Kendall tau clustermap of participants (blacklist marked)
  step5_models_humans               same, models included
  step5_model_vs_human.csv          Kendall tau-b: model vs average human, model vs best individual, human ceiling
"""
import argparse, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau, wilcoxon, mannwhitneyu
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from src.config import RESULTS, MODEL_ORDER
from src.metrics import P_COLS as P
# same scoring, exclusions and balanced threshold as the human analysis
from human_analysis import load_long, check_participants, MAX_FAILED_CHECKS, BALANCED_HALF_WIDTH as BALANCED, HUMAN_DIR
from human_figures import save, TEXT_W   # same figure style/size as the human figures (sets rcParams)

CATS = ["Ins/Pos", "Mtx/Emb", "Ser/Mul"]
CONTROL_OK = 0.90          # step 1: flag model x category below this control accuracy
C_THRESHOLD = 0.10         # 'real' contradiction mass
FAM_COLOR = {"classifier": "#2a78d6", "generative": "#eb6834"}
# self-reported English proficiency (human_prep.py), one ordered purple ramp; anything else = grey
PROF_COLOR = {"Native English speaker": "#3f2a8c", "C2 - Proficient": "#7b5fc4", "C1 - Advanced": "#b7a4e3",
              "B2 - Upper Intermediate": "#d9cdf0", "B1 - Intermediate": "#eee8f8"}
PROF_OTHER = "#8c8c8c"


def _wil(x):
    """Wilcoxon signed-rank p-value against 0 (NaN if all values are 0 or too few)."""
    x = pd.Series(x).dropna()
    return wilcoxon(x).pvalue if (x != 0).sum() > 0 else np.nan


# ----------------------------------------------------------------------------- data
def load_models(path):
    s = pd.read_csv(path)
    if "section" not in s.columns:
        s["section"] = s["item_id"].str[0]
    # one row per model x item x slot, LLMs averaged over the prompt variants
    m = (s.groupby(["model", "family", "item_id", "category", "slot"], as_index=False)[P].mean())
    present = list(dict.fromkeys(s["model"]))
    order = [mo for mo in MODEL_ORDER if mo in present] + [mo for mo in present if mo not in MODEL_ORDER]
    return s, m, order


def wide(m, col):
    return m.pivot_table(index=["model", "family", "item_id", "category"], columns="slot", values=col)


def load_humans(key_path, v1, v2):
    key = pd.read_csv(key_path)
    L = pd.concat([load_long(key, v1, 1), load_long(key, v2, 2)], ignore_index=True)
    part = check_participants(L)
    black = set(part.index[part["excluded"]])
    items = L[L["kind"] == "item"]
    H = items.pivot_table(index="item_id", columns="participant", values="human_lean")
    # participant ids follow the row order of each cleaned response file, as in load_long
    prof = pd.Series({f"v{v}_p{i + 1}": p for v, path in ((1, v1), (2, v2))
                      for i, p in enumerate(pd.read_csv(path)["proficiency"])})
    return H, black, part["failed"], prof


# ----------------------------------------------------------------------------- step 1
def step1(m):
    pe = wide(m, "p_entailment")
    out = pe.index.to_frame(index=False)
    # pairwise: the entailed hypothesis gets higher P(E) than the non-entailed one (strict >, no tie rule)
    out["H_pair"] = (pe["ctrlH_ent"] > pe["ctrlH_non"]).to_numpy()
    out["L_pair"] = (pe["ctrlL_ent"] > pe["ctrlL_non"]).to_numpy()
    # strict: argmax label is right on all four control pairs (ent -> E, non -> not E)
    am = {}
    for slot in ["ctrlH_ent", "ctrlH_non", "ctrlL_ent", "ctrlL_non"]:
        sub = m[m["slot"] == slot].set_index(["model", "family", "item_id", "category"])[P]
        am[slot] = sub.idxmax(axis=1).reindex(pe.index)
    out["H_strict"] = ((am["ctrlH_ent"] == "p_entailment") & (am["ctrlH_non"] != "p_entailment")).to_numpy()
    out["L_strict"] = ((am["ctrlL_ent"] == "p_entailment") & (am["ctrlL_non"] != "p_entailment")).to_numpy()
    t = (out.groupby(["model", "family", "category"])
            .agg(acc_H=("H_pair", "mean"), acc_L=("L_pair", "mean"),
                 strict_H=("H_strict", "mean"), strict_L=("L_strict", "mean"), n=("item_id", "size"))
            .reset_index())
    t["passes"] = (t["acc_H"] >= CONTROL_OK) & (t["acc_L"] >= CONTROL_OK)
    return t


# ----------------------------------------------------------------------------- step 2
def step2(m):
    rows = []
    for (model, fam, cat), g in m.groupby(["model", "family", "category"]):
        r = dict(model=model, family=fam, category=cat)
        for name, slots in [("ctrl_ent", ["ctrlH_ent", "ctrlL_ent"]),
                            ("ctrl_non", ["ctrlH_non", "ctrlL_non"]),
                            ("amb", ["amb_high", "amb_low"])]:
            x = g[g["slot"].isin(slots)]["p_contradiction"]
            r[f"meanC_{name}"] = x.mean()
            r[f"shareC>={C_THRESHOLD}_{name}"] = (x >= C_THRESHOLD).mean()
        a = g[g["slot"].isin(["amb_high", "amb_low"])].pivot_table(index="item_id", columns="slot", values=P)
        eH, eL = a[("p_entailment", "amb_high")], a[("p_entailment", "amb_low")]
        otherH = a[[("p_neutral", "amb_high"), ("p_contradiction", "amb_high")]].max(axis=1)
        otherL = a[[("p_neutral", "amb_low"), ("p_contradiction", "amb_low")]].max(axis=1)
        r["n_items"] = len(a)
        r["neither_entailed"] = int(((eH < otherH) & (eL < otherL)).sum())   # README note: both E below N or C
        # how the losing reading is treated: rejected as false (C) or left open (N)
        low_wins = eL > eH
        lose_N = np.where(low_wins, a[("p_neutral", "amb_high")], a[("p_neutral", "amb_low")])
        lose_C = np.where(low_wins, a[("p_contradiction", "amb_high")], a[("p_contradiction", "amb_low")])
        r["loser_more_C_than_N"] = float((lose_C > lose_N).mean())
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- step 3 / 4
def item_table(m):
    pe = wide(m, "p_entailment")
    t = pe.index.to_frame(index=False)
    t["bias"] = (pe["amb_low"] - pe["amb_high"]).to_numpy()
    t["winner_pE"] = np.maximum(pe["amb_low"], pe["amb_high"]).to_numpy()
    H = np.stack([wide(m, c)["amb_high"].to_numpy() for c in P], 1)
    L = np.stack([wide(m, c)["amb_low"].to_numpy() for c in P], 1)
    H = np.clip(H, 0, None); L = np.clip(L, 0, None)
    H /= H.sum(1, keepdims=True); L /= L.sum(1, keepdims=True)
    t["jsd"] = [jensenshannon(h, l, base=2) for h, l in zip(H, L)]
    return t


def bias_summary(t):
    return (t.groupby(["model", "family", "category"])
              .agg(n=("bias", "size"), median_bias=("bias", "median"), mean_bias=("bias", "mean"),
                   q1=("bias", lambda x: x.quantile(.25)), q3=("bias", lambda x: x.quantile(.75)),
                   share_low_wins=("bias", lambda x: (x > 0).mean()),
                   median_gap=("bias", lambda x: x.abs().median()),
                   winner_below_half=("winner_pE", lambda x: (x < 0.5).mean()),
                   median_jsd=("jsd", "median"))
              .reset_index())


def boxplot(t, col, order, flagged, title, out, name, xlabel, xlim=None, zero=True):
    models = [mo for mo in order if mo in set(t["model"])]
    fam = t.drop_duplicates("model").set_index("model")["family"]
    cats = [c for c in CATS if c in set(t["category"])]
    fig, axs = plt.subplots(1, len(cats), figsize=(TEXT_W, 0.25 * len(models) + 1.2), sharey=True,
                            squeeze=False)
    axs = axs[0]
    for ax, cat in zip(axs, cats):
        sub = t[t["category"] == cat]
        data = [sub[sub["model"] == mo][col].dropna().to_numpy() for mo in models]
        bp = ax.boxplot(data, orientation="horizontal", widths=0.6, patch_artist=True, showfliers=True,
                        flierprops=dict(marker="o", markersize=2, alpha=0.4, markeredgecolor="#666"),
                        medianprops=dict(color="#222", lw=1.2), whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8),
                        boxprops=dict(lw=0.8))
        for patch, mo in zip(bp["boxes"], models):
            patch.set_facecolor(FAM_COLOR[fam[mo]]); patch.set_alpha(0.55); patch.set_edgecolor("#444")
            if (mo, cat) in flagged:
                patch.set_hatch("///"); patch.set_alpha(0.25)
        if zero: ax.axvline(0, color="#999", lw=0.8, ls="--")
        ax.set_title(cat, fontweight="bold")
        ax.set_xlabel(xlabel)
        if xlim: ax.set_xlim(*xlim)
        ax.grid(axis="x", color="#eee"); ax.set_axisbelow(True)
    axs[0].set_yticks(range(1, len(models) + 1)); axs[0].set_yticklabels(models)
    axs[0].invert_yaxis()
    from matplotlib.patches import Patch
    fam_label = {"classifier": "NLI classifier", "generative": "generative LLM"}
    hand = [Patch(facecolor=FAM_COLOR[f], alpha=.55, label=fam_label[f]) for f in FAM_COLOR if f in set(fam)]
    hand += [Patch(facecolor="white", edgecolor="#444", hatch="///", label="fails controls (<90%)")]
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    fig.legend(handles=hand, loc="upper center", ncol=len(hand), frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(fig, Path(out), name)


# ----------------------------------------------------------------------------- step 5
def tau(a, b):
    ok = a.notna() & b.notna()
    if ok.sum() < 3: return np.nan
    return kendalltau(a[ok], b[ok]).statistic   # tau-b (handles ties)


def style_cbar(g, legend, gap=0.03):
    """Small tick labels; colorbar and the strip legend sit just right of the row labels, whatever their length.
    legend: list of (header, {label: color}) for the colour strips."""
    from matplotlib.patches import Patch
    g.ax_heatmap.tick_params(labelsize=6)
    g.ax_cbar.tick_params(labelsize=6)
    g.ax_row_colors.tick_params(labelsize=6)
    g.fig.canvas.draw()
    inv = g.fig.transFigure.inverted()
    right = max(inv.transform((t.get_window_extent().x1, 0))[0] for t in g.ax_heatmap.get_yticklabels())
    g.ax_cbar.set_position([right + gap, 0.45, 0.02, 0.25])
    hand = []
    for header, colors in legend:
        hand.append(Patch(visible=False, label=header))
        hand += [Patch(facecolor=c, edgecolor="#999", lw=0.3, label=lab) for lab, c in colors.items()]
    leg = g.fig.legend(handles=hand, loc="upper left", bbox_to_anchor=(right + gap - 0.01, 0.4),
                       frameon=False, fontsize=6, handlelength=1, handleheight=1)
    for txt in leg.get_texts():
        if txt.get_text() in {h for h, _ in legend}:
            txt.set_fontweight("bold"); txt.set_x(-12)


def prof_strip(index, prof):
    """Proficiency colour per row (white for models) and the legend entries for the levels present."""
    colors = {p: PROF_COLOR.get(prof[p], PROF_OTHER) if p in prof.index else "white" for p in index}
    levels = [lv for lv in PROF_COLOR if lv in set(prof.reindex(index).dropna())]
    entries = {lv.split(" - ")[0].replace("Native English speaker", "native"): PROF_COLOR[lv] for lv in levels}
    if any(c == PROF_OTHER for c in colors.values()):
        entries["other / unclear"] = PROF_OTHER
    return pd.Series(colors), entries


def step5(t, H, black, prof, out, order):
    keep = [p for p in H.columns if p not in black]
    avg = H[keep].mean(axis=1)
    rows = []
    # human ceiling: each kept participant vs the average of the OTHER kept participants
    cats = [c for c in CATS if c in set(t["category"])] + ["all"]
    ceil = {}
    for cat in cats:
        ids = H.index if cat == "all" else t[t["category"] == cat]["item_id"].unique()
        Hc = H.loc[H.index.intersection(ids), keep]
        loo = [tau(Hc[p], Hc.drop(columns=p).mean(axis=1)) for p in keep]
        pair = [tau(Hc[a], Hc[b]) for i, a in enumerate(keep) for b in keep[i + 1:]]
        ceil[cat] = (np.nanmean(loo), np.nanmean(pair))
    for model in [mo for mo in order if mo in set(t["model"])]:
        g = t[t["model"] == model].set_index("item_id")
        for cat in cats:
            gi = g if cat == "all" else g[g["category"] == cat]
            ids = gi.index.intersection(H.index)
            if len(ids) < 3:
                continue
            mb = gi.loc[ids, "bias"]
            indiv = {p: tau(mb, H.loc[ids, p]) for p in keep}
            valid = {p: v for p, v in indiv.items() if not np.isnan(v)}
            best = max(valid, key=valid.get) if valid else None
            rows.append(dict(model=model, category=cat, n_items=len(ids),
                             tau_avg_human=tau(mb, avg.loc[ids]),
                             tau_best_individual=valid.get(best, np.nan), best_individual=best,
                             human_ceiling_loo=ceil[cat][0], human_mean_pairwise=ceil[cat][1]))
    res = pd.DataFrame(rows)
    res.to_csv(f"{out}/step5_model_vs_human.csv", index=False)

    # clustermap of participants (all, blacklist marked)
    item_cat = t.drop_duplicates("item_id").set_index("item_id")["category"]

    def kendall_matrix(df):
        # tau computed WITHIN each category, then averaged (weighted by #items), so that two raters
        # do not look similar just because both separate the categories (Ins/Pos High vs Mtx/Emb Low)
        cols = list(df.columns)
        cats = item_cat.reindex(df.index)
        w = {c: (cats == c).sum() for c in CATS}
        M = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                vals = [(tau(df.loc[cats == c, a], df.loc[cats == c, b]), w[c]) for c in CATS if w[c] > 2]
                vals = [(v, n) for v, n in vals if not np.isnan(v)]
                M.loc[a, b] = M.loc[b, a] = (sum(v * n for v, n in vals) / sum(n for _, n in vals)) if vals else np.nan
        return M.fillna(0)

    M = kendall_matrix(H)
    ps, pleg = prof_strip(M.index, prof)
    rc = pd.DataFrame({"status": ["#d03b3b" if p in black else "#cccccc" for p in M.index],
                       "proficiency": ps}, index=M.index)
    g = sns.clustermap(M, method="average", cmap="RdBu_r", vmin=-1, vmax=1, row_colors=rc,
                       figsize=(0.85 * TEXT_W, 0.85 * TEXT_W), cbar_kws={"label": "Kendall τ-b"}, dendrogram_ratio=0.12)
    style_cbar(g, [("Status", {"kept": "#cccccc", "excluded": "#d03b3b"}),
                   ("Proficiency", pleg)])
    g.fig.suptitle(f"Participants: Kendall τ within categories, averaged (excluded = >{MAX_FAILED_CHECKS} failed checks)", y=1.01)
    save(g.fig, Path(out), "step5_human_human")

    # models + kept humans together
    mb = t.pivot_table(index="item_id", columns="model", values="bias")
    both = pd.concat([H[keep], mb], axis=1)
    M2 = kendall_matrix(both)
    ps2, pleg2 = prof_strip(M2.index, prof)
    rc2 = pd.DataFrame({"rater": ["#eb6834" if c in mb.columns else "#2a78d6" for c in M2.index],
                        "proficiency": ps2}, index=M2.index)
    g = sns.clustermap(M2, method="average", cmap="RdBu_r", vmin=-1, vmax=1, row_colors=rc2,
                       figsize=(0.85 * TEXT_W, 0.85 * TEXT_W), cbar_kws={"label": "Kendall τ-b"}, dendrogram_ratio=0.1)
    style_cbar(g, [("Rater", {"human": "#2a78d6", "model": "#eb6834"}), ("Proficiency", pleg2)])
    g.fig.suptitle("Humans and models: Kendall τ within categories, averaged", y=1.01)
    save(g.fig, Path(out), "step5_models_humans")
    return res, avg


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=RESULTS, help="long model scores from eval_models.py")
    ap.add_argument("--key", default=HUMAN_DIR / "key.csv")
    ap.add_argument("--v1", default=HUMAN_DIR / "v1.csv", help="output of human_prep.py")
    ap.add_argument("--v2", default=HUMAN_DIR / "v2.csv", help="output of human_prep.py")
    ap.add_argument("--no-humans", action="store_true", help="skip the human steps (3b, 5)")
    ap.add_argument("--out", default=ROOT / "results" / "analysis")
    ap.add_argument("--balanced", type=float, default=BALANCED)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    s, m, order = load_models(a.models)

    s1 = step1(m); s1.to_csv(f"{a.out}/step1_controls.csv", index=False)
    flagged = set(zip(s1.loc[~s1["passes"], "model"], s1.loc[~s1["passes"], "category"]))
    print("\n== STEP 1: control accuracy (pairwise; flag < 0.90) ==")
    print(s1.round(2).to_string(index=False))

    s2 = step2(m); s2.to_csv(f"{a.out}/step2_contradiction.csv", index=False)
    print("\n== STEP 2: contradiction ==")
    print(s2.round(2).to_string(index=False))

    t = item_table(m)
    s3 = bias_summary(t); s3.to_csv(f"{a.out}/step3_bias.csv", index=False)
    print("\n== STEP 3: bias = P(E|Low) - P(E|High) ==")
    print(s3.drop(columns="median_jsd").round(2).to_string(index=False))
    boxplot(t, "bias", order, flagged, "Attachment bias per item: P(E | Low reading) − P(E | High reading)   (> 0 = prefers Low)",
            a.out, "fig_step3_bias", "P(E|Low) − P(E|High)", xlim=(-1.05, 1.05))

    s4 = t.groupby(["model", "family", "category"])["jsd"].describe().reset_index()
    s4.to_csv(f"{a.out}/step4_jsd.csv", index=False)
    boxplot(t, "jsd", order, flagged, "JS distance between the E/N/C distributions of the High and Low readings",
            a.out, "fig_step4_jsd", "JS distance (0 = identical)", xlim=(-0.02, 1.02), zero=False)
    print("\n== STEP 4: JSD (median per model x category) ==")
    print(t.pivot_table(index="model", columns="category", values="jsd", aggfunc="median").reindex(order).round(2).to_string())

    if not a.no_humans:
        H, black, failed, prof = load_humans(a.key, a.v1, a.v2)
        keep = [p for p in H.columns if p not in black]
        lean = H[keep].mean(axis=1)
        se = H[keep].std(axis=1) / np.sqrt(H[keep].notna().sum(axis=1))
        # MAIN: items without a significant human lean (95% CI of the mean lean includes 0)
        # ROBUSTNESS: items with a small human lean (|mean lean| < --balanced)
        subsets = {"ns": (set(lean.index[(lean - 1.96 * se <= 0) & (lean + 1.96 * se >= 0)]),
                          "items without a significant human lean (95% CI includes 0)"),
                   "balanced": (set(lean.index[lean.abs() < a.balanced]),
                                f"items humans find balanced (|mean lean| < {a.balanced})")}
        for name, (ids, label) in subsets.items():
            tb = t[t["item_id"].isin(ids)]
            sb = bias_summary(tb); sb.to_csv(f"{a.out}/step3_bias_{name}.csv", index=False)
            boxplot(tb, "bias", order, flagged,
                    f"Only {label}: " + ", ".join(f"{c} {tb[tb.category == c].item_id.nunique()}"
                                                  for c in CATS if c in set(t["category"])) + " items",
                    a.out, f"fig_step3_bias_{name}", "P(E|Low) − P(E|High)", xlim=(-1.05, 1.05))
            rows = []
            for (mo, c), g in t.groupby(["model", "category"]):
                sub, rest = g[g["item_id"].isin(ids)]["bias"], g[~g["item_id"].isin(ids)]["bias"]
                rows.append(dict(model=mo, category=c, n_all=len(g), median_all=g["bias"].median(),
                                 p_all=_wil(g["bias"]), n_sub=len(sub), median_sub=sub.median(),
                                 p_sub=_wil(sub), median_rest=rest.median(),
                                 p_sub_vs_rest=mannwhitneyu(sub, rest).pvalue if len(sub) and len(rest) else np.nan))
            cmp = pd.DataFrame(rows)
            cmp.to_csv(f"{a.out}/step3_all_vs_{name}.csv", index=False)
            print(f"\n== STEP 3b ({'MAIN' if name == 'ns' else 'robustness'}): all items vs {label} ==")
            print(cmp.round(3).to_string(index=False))

        res, _ = step5(t, H, black, prof, a.out, order)
        print(f"\n== STEP 5: Kendall tau-b (participants: {len(H.columns)}, excluded: {sorted(black)}) ==")
        print(res.round(3).to_string(index=False))
    print(f"\nall outputs in {a.out}/")


if __name__ == "__main__":
    main()
