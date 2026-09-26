import pytest

from polystep_or import Tuner

SPACE = {"x": ("lin", -2.0, 2.0), "y": ("log", 0.01, 10.0), "flag": [False, True]}


def quadratic(config, seed):
    return (config["x"] - 0.5) ** 2 + (config["y"] - 1.0) ** 2 + (0.0 if config["flag"] else 0.1)


def test_sobol_finds_the_minimum_region():
    best, table = Tuner(SPACE, n_trials=64, seeds=[1, 2]).run(quadratic)
    assert len(table) == 64
    assert all(set(row["values"]) == {1, 2} for row in table)
    assert best == min(table, key=lambda row: row["mean"])["config"]
    assert quadratic(best, 0) < 0.3


def test_sobol_design_is_reproducible_and_rounded():
    first = Tuner(SPACE, n_trials=8, digits=4).configs()
    second = Tuner(SPACE, n_trials=8, digits=4).configs()
    assert first == second
    assert all(round(c["x"], 4) == c["x"] for c in first)
    assert all(-2.0 <= c["x"] <= 2.0 and 0.01 <= c["y"] <= 10.0 for c in first)


def test_store_continues_without_calling_the_objective(tmp_path):
    tuner = Tuner(SPACE, n_trials=4, seeds=[0], store=tmp_path)
    best, _ = tuner.run(quadratic)

    def fail(config, seed):
        raise AssertionError("objective called although every unit is stored")

    assert tuner.run(fail)[0] == best
    assert len(list(tmp_path.glob("trial*_seed0.json"))) == 4


def test_parallel_units_and_reading_partial_results(tmp_path):
    tuner = Tuner(SPACE, n_trials=4, seeds=[0, 1], store=tmp_path)
    tuner.run(quadratic, trials=[2])
    best, table = tuner.run(None)
    assert [row["mean"] is not None for row in table] == [False, False, True, False]
    assert best == table[2]["config"]


def test_grid_and_optuna():
    grid = {"x": [-1.0, 0.5, 2.0], "y": [1.0, 3.0], "flag": [True]}
    best, table = Tuner(grid, strategy="grid").run(quadratic)
    assert len(table) == 6 and best == {"x": 0.5, "y": 1.0, "flag": True}
    best, table = Tuner(SPACE, strategy="optuna", n_trials=12).run(quadratic)
    assert len(table) == 12 and quadratic(best, 0) <= min(r["mean"] for r in table) + 1e-12


def test_invalid_arguments():
    with pytest.raises(ValueError):
        Tuner(SPACE, strategy="random")
    with pytest.raises(ValueError):
        Tuner(SPACE, strategy="grid").configs()
