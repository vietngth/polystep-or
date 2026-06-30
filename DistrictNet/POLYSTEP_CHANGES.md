# DistrictNet: changes relative to upstream

Upstream: `https://github.com/cheikh025/DistrictNet`, cloned at commit `7a624a3`. This directory is that clone with a small adapter layer that adds PolyStep as an estimator and hardens the large-instance solve. The exact line-level diff for every modified file is in `POLYSTEP_CHANGES.patch`.

## Files added by us (new, not in upstream)

- `src/Estimators/polyStep.jl` : the PolyStep estimator. It keeps DistrictNet's solver but discards the optimal-partition label and the Fenchel-Young smoothing, and instead minimizes the realized simulated routing cost directly with the gradient-free PolyStep update. This is the only learning code we add; the optimizer itself is imported unmodified from the PolyStep library.
- `trainatscale_driver.jl` : trains PolyStep label-free at the target-20 deployment scale on several real cities, the regime where DistrictNet's exact label is intractable, then evaluates against the small-trained models.
- `eval_driver.jl`, `datahelp_eval.jl`, `repair_eval.jl`, `feasibility_sidebyside.jl`, `dump_perdistrict.jl`, `time_districtnet.jl`, `test_repair.jl` : evaluation and figure drivers. They call the upstream solver and the shared out-of-sample SAA simulator; they do not change the method.

## Files modified (adapter and feasibility hardening)

- `src/districtNet.jl`, `experiments.jl`, `experiment_evaluator.jl` : register the PolyStep estimator and the train-at-scale entry point alongside the existing methods, and route every method through the same shared-scenario out-of-sample SAA evaluation so the comparison is matched.
- `src/Solver/Kruskal.jl` : a bidirectional cascade chain-repair so the large instance (Ile-de-France, 2000 units, target 20) yields a feasible districting at the full district count. The upstream repair could leave over-max-saturated or under-min orphan districts.
- `src/district.jl`, `src/Solver/exactsolver.jl` : a guard for disconnected or empty districts so the spanning-tree cost and the size-deviation term are well defined under the deploy-scale solver.
- `src/learning.jl` : hooks so the realized-cost trainer can drive the existing model module.
- `src/Estimators/{AvgTsp,BD,FIG,predictGnn}.jl` : align the reference baselines to the same shared scenarios and seed used by the evaluation, so all methods are scored on identical demand draws.

## What is unchanged

The graph neural network, the constrained-minimum-spanning-tree surrogate, the perturbed-optimizer Fenchel-Young training of DistrictNet itself, the exact set-partitioning solver at small scale, and the C++ routing-cost simulator are upstream verbatim. PolyStep reuses the same solver and simulator and adds only the realized-cost training loop.

One configuration caveat: to fit the comparison into the cluster budget, the runs reduce several compute limits below the upstream defaults. Solver and ILS `MAX_TIME` drop from 1200 s to 120 s across the estimators, the GNN epoch counts are cut (predictGnn 1000 to 40, districtNet 100 to 40), and the evaluation `NB_SCENARIO` is set to 50. These are budget choices, not method changes; the architectures, the Fenchel-Young objective, the CMST forward solver, and the data are the authors' originals.
