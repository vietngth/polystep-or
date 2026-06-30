"""Driver (non-destructive): visualize the tree+PolyStep predictor as a LEFT-TO-RIGHT
PIPELINE on the PyEPO shortest-path grid, so the decision unfolds stage by stage:
  (1) input features x  ->  (2) tree rule (root->leaf flowchart)  ->
  (3) chosen leaf's predicted edge costs  ->  (4) shortest path under those costs.
Saves figures/tree_decisions.pdf (vector). Same trained tree+PolyStep model and 5x5 grid."""
import sys, os, numpy as np, torch
sys.path.insert(0, "polystep/src")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

from interpretable_predictor import HardTree, fit_balanced_cart, route
from pto.capability import setup_sp, train_two_stage, dev, PF
from pyepo.data import shortestpath

# ---- Okabe-Ito colourblind-safe palette ----
OI = dict(orange="#E69F00", skyblue="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", vermillion="#D55E00", purple="#CC79A7", black="#000000")

SEED, DEG, DEPTH, H, W = 0, 4, 3, 5, 5
PS_STEPS = 80

# ---------------------------------------------------------------- train (unchanged model)
cfg, _ = setup_sp(SEED, DEG, H=H, W=W)
ts = train_two_stage(cfg); cfg["warm"] = ts

# warm-start CART, then refine a few PolyStep steps (mirrors train_polystep_tree)
m = HardTree(PF, cfg["dim"], DEPTH, SEED).to(dev)
X = cfg["Xtr"].cpu().numpy(); Ctr_std = cfg["Cs"].cpu().numpy()
fl_np, thr_np0, leaf_np = fit_balanced_cart(X, Ctr_std, DEPTH)
with torch.no_grad():
    m.fl.copy_(torch.tensor(fl_np, device=dev)); m.thr.copy_(torch.tensor(thr_np0, device=dev))
    m.leaf.copy_(torch.tensor(leaf_np, device=dev))

from polystep import PolyStepOptimizer
from polystep.epsilon import CosineEpsilon
pso = PolyStepOptimizer(m, polytope_type="orthoplex", epsilon=CosineEpsilon(0.5, 0.05),
                        step_radius=0.4, probe_radius=0.8, num_probe=1, seed=SEED,
                        use_momentum=True, momentum_init=0.5, momentum_final=0.9)
Xtr, Cs, solve, sgn = cfg["Xtr"], cfg["Cs"], cfg["ps_solve"], cfg["sign"]
def closure(bp):
    pred = route(bp["fl"], bp["thr"], bp["leaf"], Xtr); N, B, E = pred.shape
    w = solve(pred.reshape(N * B, E)).reshape(N, B, E)
    return sgn * (w * Cs.unsqueeze(0)).sum(-1).mean(-1)
for _ in range(PS_STEPS):
    pso.step(closure)

# ---------------------------------------------------------------- denormalize stats
# reproduce the affine standardization used in _common so predicted costs print in real units
x_all, c_all = shortestpath.genData(900 + 200, PF, (H, W), deg=DEG, noise_width=0, seed=SEED)
Ctr_raw = c_all[:900]
C_SHIFT = float(Ctr_raw.mean()); C_STD = float(Ctr_raw.std())

# ---------------------------------------------------------------- extract tree + route helpers
fl = m.fl.detach(); thr = m.thr.detach(); leaf = m.leaf.detach()
I = 2 ** DEPTH - 1
jsel = fl.argmax(-1).cpu().numpy()             # (I,) feature index per internal node
thr_np = thr.cpu().numpy()                     # (I,) threshold per internal node
leaf_np = leaf.cpu().numpy()                   # (L,E) standardized predicted edge costs

arcs = list(cfg["om"].arcs)
Xte = cfg["Xte"]; Cte = cfg["Cte"]
Xte_np = Xte.cpu().numpy()


def trace(xrow):
    """Hard root-to-leaf routing. Returns (leaf_idx, steps) with steps =
    [(node, feature j, threshold t, go in {0,1}, x[j]) ...]."""
    cur = 0; steps = []
    for _ in range(DEPTH):
        j = int(jsel[cur]); t = float(thr_np[cur]); xv = float(xrow[j])
        go = 1 if xv > t else 0
        steps.append((cur, j, t, go, xv))
        cur = 2 * cur + 1 + go
    return cur - I, steps


# group test instances by leaf, and cache each leaf's chosen path
leafmap = {}
for n in range(Xte_np.shape[0]):
    li, _ = trace(Xte_np[n])
    leafmap.setdefault(li, []).append(n)
chosen = {li: solve(leaf[li].unsqueeze(0))[0].cpu().numpy() for li in leafmap}     # arc indicators
w_opt_all = solve(Cte).cpu().numpy()                                               # per-instance optimum
print("Leaf populations:", {k: len(v) for k, v in leafmap.items()})

# ---------------------------------------------------------------- pick ONE clear representative
# Prefer a well-populated leaf whose root-to-leaf path mixes yes/no branches (so the flowchart
# shows both edge types) and whose chosen route differs from the true optimum by a small,
# visible regret (so panel (4) is illustrative). Fall back to the most populated leaf.
def score(li):
    inst = leafmap[li][0]
    _, steps = trace(Xte_np[inst])
    c_true = Cte[inst].cpu().numpy()
    realized = float((chosen[li] * c_true).sum())
    optimal = float((w_opt_all[inst] * c_true).sum())
    regret = realized - optimal
    mixed = len({s[3] for s in steps}) > 1
    ok_reg = 0.08 < regret < 1.5
    return (mixed, ok_reg, len(leafmap[li]), -regret), inst, steps, realized, optimal, regret

cands = {li: score(li) for li in leafmap}
li = max(cands, key=lambda k: cands[k][0])
_, inst, steps, realized, optimal, regret = cands[li]
print(f"Picked leaf {li}, test instance {inst}: realized={realized:.3f} "
      f"optimal={optimal:.3f} regret={regret:.3f}")

xrow = Xte_np[inst]
pc_real = leaf_np[li] * C_STD + C_SHIFT          # predicted edge costs in real units
w_chosen = chosen[li]
w_opt = w_opt_all[inst]
active_nodes = [s[0] for s in steps]             # internal nodes on the active path
used_feats = sorted({s[1] for s in steps})       # features the active rule reads

# ================================================================ figure scaffold
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(12.8, 4.2))
# columns: features | arrow | tree | arrow | cost-grid | colorbar | arrow | route-grid
gs = fig.add_gridspec(1, 8, width_ratios=[1.0, 0.36, 1.88, 0.36, 1.18, 0.14, 0.66, 1.22],
                      left=0.035, right=0.985, top=0.83, bottom=0.07, wspace=0.0)
ax_feat = fig.add_subplot(gs[0, 0])
ax_a1 = fig.add_subplot(gs[0, 1])
ax_tree = fig.add_subplot(gs[0, 2])
ax_a2 = fig.add_subplot(gs[0, 3])
ax_cost = fig.add_subplot(gs[0, 4])
ax_cbar = fig.add_subplot(gs[0, 5])
ax_a3 = fig.add_subplot(gs[0, 6])
ax_route = fig.add_subplot(gs[0, 7])

HL = OI["orange"]        # highlight colour for the active rule / path
DIM = "#C7C7C7"


def arrow(ax, label):
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.annotate("", xy=(0.90, 0.50), xytext=(0.10, 0.50),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color=OI["black"],
                                shrinkA=0, shrinkB=0))
    ax.text(0.5, 0.585, label, ha="center", va="bottom", fontsize=8.0,
            color="#333333", style="italic")

arrow(ax_a1, "apply\ntree rule")
arrow(ax_a2, "read\nleaf costs")
arrow(ax_a3, "solve\nshortest path")

# ---------------------------------------------------------------- (1) features
xv = xrow.astype(float)
cols = [HL if k in used_feats else DIM for k in range(len(xv))]
ax_feat.bar(range(len(xv)), xv, color=cols, edgecolor=OI["black"], linewidth=0.6, width=0.7)
ax_feat.axhline(0, color=OI["black"], lw=0.8)
ax_feat.set_xticks(range(len(xv)))
ax_feat.set_xticklabels([f"x{k}" for k in range(len(xv))], fontsize=9)
ax_feat.set_title("(1) features  x", fontsize=10.5, fontweight="bold")
ax_feat.set_ylabel("value", fontsize=9)
ax_feat.tick_params(labelsize=8)
for s in ("top", "right"):
    ax_feat.spines[s].set_visible(False)
mx = np.abs(xv).max() * 1.28
ax_feat.set_ylim(-mx, mx)
for k, v in enumerate(xv):
    ax_feat.annotate(f"{v:+.2f}", (k, v), ha="center", va="bottom" if v >= 0 else "top",
                     fontsize=6.8, xytext=(0, 2 if v >= 0 else -2), textcoords="offset points")
ax_feat.text(0.5, -0.165, "orange = read by the rule", transform=ax_feat.transAxes,
             ha="center", va="top", fontsize=7.4, color=HL)

# ---------------------------------------------------------------- (2) tree rule (flowchart)
ax_tree.set_xlim(0, 1); ax_tree.set_ylim(0, 1); ax_tree.axis("off")
ax_tree.set_title("(2) tree rule  (root → leaf)", fontsize=10.5, fontweight="bold")

LVL_Y = {0: 0.90, 1: 0.64, 2: 0.38}
LEAF_Y = 0.13


def node_pos(n):
    lvl = int(np.floor(np.log2(n + 1))); pos = n + 1 - 2 ** lvl
    return (pos + 0.5) / (2 ** lvl), LVL_Y[lvl]


def leaf_pos(k):
    return (k + 0.5) / 8.0, LEAF_Y


# active edges (parent -> chosen child)
active_edge = set()
for (n, j, t, go, xvv) in steps:
    active_edge.add((n, 2 * n + 1 + go))

# draw edges (faint grey, active in highlight colour)
for n in range(I):
    xn, yn = node_pos(n)
    for child in (2 * n + 1, 2 * n + 2):
        xc, yc = node_pos(child) if child < I else leaf_pos(child - I)
        on = (n, child) in active_edge
        ax_tree.plot([xn, xc], [yn, yc], "-", color=HL if on else DIM,
                     lw=2.6 if on else 1.0, zorder=2 if on else 1, solid_capstyle="round")

# internal node boxes ("x[j] > tau ?")
for n in range(I):
    xn, yn = node_pos(n)
    on = n in active_nodes
    txt = f"x{jsel[n]} > {thr_np[n]:.2f}?"
    box = FancyBboxPatch((xn - 0.088, yn - 0.052), 0.176, 0.104,
                         boxstyle="round,pad=0.006,rounding_size=0.02",
                         fc="#FCEFD7" if on else "white", ec=HL if on else "#9A9A9A",
                         lw=1.8 if on else 1.0, zorder=3, mutation_aspect=0.55)
    ax_tree.add_patch(box)
    ax_tree.text(xn, yn, txt, ha="center", va="center", fontsize=7.1,
                 fontweight="bold" if on else "normal",
                 color=OI["black"] if on else "#555555", zorder=4)

# leaf squares (chosen leaf highlighted + filled)
for k in range(8):
    xk, yk = leaf_pos(k); on = (k == li)
    sz = 0.052 if on else 0.036
    ax_tree.add_patch(FancyBboxPatch((xk - sz, yk - sz), 2 * sz, 2 * sz,
                                     boxstyle="square,pad=0", fc=HL if on else "white",
                                     ec=HL if on else "#9A9A9A", lw=1.8 if on else 1.0, zorder=3))
    ax_tree.text(xk, yk - 0.075, f"L{k}", ha="center", va="top", fontsize=6.4,
                 color=OI["black"] if on else "#888888",
                 fontweight="bold" if on else "normal")

# yes/no labels on the active edges
for (n, j, t, go, xvv) in steps:
    child = 2 * n + 1 + go
    xn, yn = node_pos(n); xc, yc = node_pos(child) if child < I else leaf_pos(child - I)
    ax_tree.text((xn + xc) / 2 + (0.028 if go else -0.028), (yn + yc) / 2,
                 "yes" if go else "no", ha="center", va="center", fontsize=7.0,
                 color=HL, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none"), zorder=5)
ax_tree.text(0.5, -0.03, "left edge = no    •    right edge = yes", transform=ax_tree.transAxes,
             ha="center", va="top", fontsize=7.4, color="#555555")

# ---------------------------------------------------------------- grids (3) & (4)
def node_xy(v):
    i, jc = divmod(v, W)
    return jc, (H - 1 - i)                                  # row 0 at top

segs = [(node_xy(u), node_xy(v)) for (u, v) in arcs]
vmin, vmax = float(pc_real.min()), float(pc_real.max())
cmap = plt.get_cmap("cividis")

# (3) predicted edge costs, NO paths
ax_cost.add_collection(LineCollection(segs, cmap=cmap, array=pc_real, linewidths=5.5,
                                      norm=plt.Normalize(vmin, vmax), alpha=0.95,
                                      capstyle="round", zorder=1))
for v in range(H * W):
    x, y = node_xy(v); ax_cost.plot(x, y, "o", ms=3.8, color=OI["black"], zorder=3)
ax_cost.set_title(f"(3) predicted edge costs\n(chosen leaf L{li})", fontsize=10.5, fontweight="bold")
ax_cost.set_xlim(-0.5, W - 0.5); ax_cost.set_ylim(-0.6, H - 0.4)
ax_cost.set_aspect("equal"); ax_cost.axis("off")

# dedicated colorbar axis, shrunk to match the grid height (no overlap with the arrow)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax)); sm.set_array([])
cpos = ax_cost.get_position(); bpos = ax_cbar.get_position()
ax_cbar.set_position([bpos.x0, cpos.y0 + 0.16 * cpos.height, bpos.width * 0.55, cpos.height * 0.60])
cb = fig.colorbar(sm, cax=ax_cbar)
cb.set_label("predicted cost", fontsize=8); cb.ax.tick_params(labelsize=7)

# (4) chosen route (bold) over faint true-optimal route
ax_route.add_collection(LineCollection(segs, colors="#DDDDDD", linewidths=2.0,
                                       capstyle="round", zorder=1))
opt_segs = [segs[e] for e in range(len(arcs)) if w_opt[e] > 0.5]
ax_route.add_collection(LineCollection(opt_segs, colors=OI["blue"], linewidths=7.0,
                                       alpha=0.32, capstyle="round", zorder=2))
ch_segs = [segs[e] for e in range(len(arcs)) if w_chosen[e] > 0.5]
ax_route.add_collection(LineCollection(ch_segs, colors=OI["vermillion"], linewidths=3.4,
                                       capstyle="round", zorder=3))
for v in range(H * W):
    x, y = node_xy(v); ax_route.plot(x, y, "o", ms=3.8, color=OI["black"], zorder=4)
sx, sy = node_xy(0); tx, ty = node_xy(H * W - 1)
ax_route.annotate("S", (sx, sy), textcoords="offset points", xytext=(-11, 4),
                  fontsize=11, fontweight="bold")
ax_route.annotate("T", (tx, ty), textcoords="offset points", xytext=(5, -12),
                  fontsize=11, fontweight="bold")
ax_route.set_title(f"(4) chosen route\nrealized = {realized:.2f}   regret = {regret:.2f}",
                   fontsize=10.5, fontweight="bold")
ax_route.set_xlim(-0.5, W - 0.5); ax_route.set_ylim(-0.6, H - 0.4)
ax_route.set_aspect("equal"); ax_route.axis("off")
ax_route.legend(handles=[Line2D([0], [0], color=OI["vermillion"], lw=3.4, label="chosen"),
                         Line2D([0], [0], color=OI["blue"], lw=7.0, alpha=0.32, label="true optimum")],
                loc="lower center", ncol=2, frameon=False, fontsize=7.6,
                bbox_to_anchor=(0.5, -0.16), handlelength=1.4, columnspacing=1.0)

fig.suptitle("Tree+PolyStep decision pipeline  (shortest path on a 5×5 grid)",
             fontsize=13, fontweight="bold", y=0.975)

os.makedirs("iclr2027-submission/figures", exist_ok=True)
out = "iclr2027-submission/figures/tree_decisions.pdf"
fig.savefig(out)                                  # vector PDF
print("SAVED", os.path.abspath(out))
print(f"INSTANCE leaf L{li}, test#{inst}: rule = "
      + " → ".join(f"x{j}{'>' if go else '≤'}{t:+.2f}" for (_, j, t, go, _) in steps)
      + f" → L{li} | realized={realized:.3f} optimal={optimal:.3f} regret={regret:.3f}")
