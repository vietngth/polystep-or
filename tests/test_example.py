from examples.new_problem import PredictedWeightKnapsack, main


def test_recourse_restores_feasibility():
    problem = PredictedWeightKnapsack(sizes=(4, 4, 4))
    _, weights = problem.load_data("test")
    everything = weights.new_ones(weights.shape)
    kept = problem.recourse(everything, weights)
    assert bool(((kept * weights).sum(-1) <= 2.0).all())
    assert bool((problem.cost(everything, weights) > problem.cost(kept, weights)).all())


def test_example_runs_end_to_end(capsys):
    main()
    out = capsys.readouterr().out
    assert "test cost: untrained" in out and "trained" in out
    trained = float(out.split("trained ")[2].split(",")[0])
    untrained = float(out.split("untrained ")[1].split(",")[0])
    assert trained < untrained
