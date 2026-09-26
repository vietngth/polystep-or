"""Named PolyStep configurations and the optimizer they build."""

from dataclasses import dataclass, fields, replace

from polystep import (
    AdaptiveSubspace,
    CosineEpsilon,
    HybridSubspace,
    LinearEpsilon,
    ParamLayout,
    PolyStepOptimizer,
)

CAP_THRESHOLD = 1024


@dataclass(frozen=True)
class PolyStepConfig:
    """Schedules, search coordinates and update options of one PolyStep run.

    Each of epsilon, step_radius and probe_radius is a pair (start, end). Equal ends give a
    constant; a constant radius multiplies the temperature. Different ends give a schedule
    over the horizon (cosine, or linear for the step radius when radius_schedule is
    "linear"); a scheduled radius is used as it is. The per-layer basis is capped at
    max_subspace_dim directions when its rank-one size exceeds CAP_THRESHOLD.
    """

    epsilon: tuple = (10.0, 0.1)
    step_radius: tuple = (5.0, 1.0)
    probe_radius: tuple = (10.0, 2.0)
    radius_schedule: str = "cosine"
    subspace: str = "hybrid"
    rank: int = 8
    max_subspace_dim: int | None = 512
    num_probe: int = 1
    momentum: bool = False
    momentum_range: tuple = (0.5, 0.9)
    scale_cost: str = "mean"
    polytope: str = "orthoplex"
    solver: str | None = "softmax"
    adaptive_num_probe: bool | None = False


CONFIGS = {
    "cosA": PolyStepConfig(epsilon=(5.0, 0.3), step_radius=(32.0, 8.0), probe_radius=(2.0, 0.5)),
    "cosB": PolyStepConfig(epsilon=(10.0, 0.1), step_radius=(5.0, 1.0), probe_radius=(10.0, 2.0)),
    "flatSNN": PolyStepConfig(epsilon=(0.5, 0.5), step_radius=(2.0, 2.0),
                              probe_radius=(1.0, 1.0), rank=4),
    "flatMoE": PolyStepConfig(epsilon=(0.5, 0.5), step_radius=(12.0, 4.0),
                              probe_radius=(1.0, 1.0), rank=4),
}

PAIRS = ("epsilon", "step_radius", "probe_radius", "momentum_range")


def get_config(config="cosB", **overrides):
    """Return a PolyStepConfig from a name, a mapping or a config, with overrides applied."""
    if isinstance(config, PolyStepConfig):
        base = config
    elif isinstance(config, str):
        if config not in CONFIGS:
            raise KeyError(f"unknown configuration {config!r}, expected one of {sorted(CONFIGS)}")
        base = CONFIGS[config]
    else:
        overrides = {**dict(config), **overrides}
        base = PolyStepConfig()
    known = {f.name for f in fields(PolyStepConfig)}
    unknown = set(overrides) - known
    if unknown:
        raise KeyError(f"unknown configuration fields {sorted(unknown)}")
    values = {k: tuple(float(x) for x in v) if k in PAIRS else v for k, v in overrides.items()}
    return replace(base, **values)


def schedule(pair, kind, horizon):
    """Return a constant for equal ends, otherwise a cosine or linear schedule."""
    start, end = (float(v) for v in pair)
    if start == end:
        return start
    if kind == "linear":
        return LinearEpsilon(init=start, target=end, decay=(start - end) / horizon)
    return CosineEpsilon(init=start, target=end, total_steps=horizon)


def make_subspace(model, config, seed):
    """Return the search subspace of a model: per-layer fixed basis or redrawn projection."""
    layout = ParamLayout.from_module(model)
    if config.subspace == "adaptive":
        return AdaptiveSubspace.from_layout(layout, rank=config.rank)
    if config.subspace != "hybrid":
        raise ValueError(f"unknown subspace {config.subspace!r}")
    rank_one = HybridSubspace.from_layout(layout, rank=1).subspace_dim
    cap = config.max_subspace_dim if rank_one > CAP_THRESHOLD else None
    return HybridSubspace.from_layout(layout, rank=config.rank, seed=seed, max_subspace_dim=cap)


def make_optimizer(model, config, horizon, seed, max_iterations=None):
    """Build a PolyStepOptimizer over the parameters of model."""
    config = get_config(config)
    horizon = max(1, int(horizon))
    kwargs = {
        "epsilon": schedule(config.epsilon, "cosine", horizon),
        "step_radius": schedule(config.step_radius, config.radius_schedule, horizon),
        "probe_radius": schedule(config.probe_radius, "cosine", horizon),
        "polytope_type": config.polytope,
        "num_probe": int(config.num_probe),
        "use_momentum": bool(config.momentum),
        "momentum_init": config.momentum_range[0],
        "momentum_final": config.momentum_range[1],
        "subspace": make_subspace(model, config, seed),
        "seed": int(seed),
        "max_iterations": int(max_iterations or horizon),
        "compile": False,
    }
    if config.scale_cost == "mean":
        kwargs["scale_cost"] = "mean"
    if config.solver is not None:
        kwargs["solver"] = config.solver
    if config.adaptive_num_probe is not None:
        kwargs["adaptive_num_probe"] = config.adaptive_num_probe
    return PolyStepOptimizer(model, **kwargs)
