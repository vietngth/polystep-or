# InferOpt_DSIRP: changes relative to upstream

Upstream: `https://github.com/tonigreif/InferOpt_DSIRP` (Greif, Bouvier, Parmentier), cloned at commit `HEAD` of the default branch. This directory is that clone with one minimal adapter change. The exact diff is in `POLYSTEP_CHANGES.patch`.

## The single change

`src/sirp_model.jl`, three added lines: a `pattern == "uniform"` branch in the instance builder. Upstream handles the `normal` and `bimodal` demand patterns; the `uniform-10` instances carry no `mean_demand` field, so the existing `start = max - mean` rule raises a `KeyError`. We set the starting inventory from the authors' own demand model: their sampler draws demand from `Uniform(0, 0.5*max)`, whose mean is `0.25*max`, so the same `start = max - mean` rule gives `start = 0.75*max`. This keeps the uniform pattern consistent with how the released code treats `normal` and `bimodal`.

## What is unchanged

Everything else is upstream verbatim and is run at the authors' paper defaults: `train_pipeline.jl` (DAgger paradigm, anticipative perfect-information IRP-MILP expert, Fenchel-Young target), the look-ahead horizon, the shortage penalty, the MILP solver selection, and `src/auxiliar.jl` (the solver builder is identical to upstream). The Toni column in the inventory-routing experiment is produced by this unmodified pipeline; PolyStep and SFGE consume only the rollout cost and need no retraining.
