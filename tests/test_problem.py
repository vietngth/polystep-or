import pytest
import torch

from polystep_or import CONFIGS, Learner, Problem, evaluate, get_config, run


class Assignment(Problem):
    """Pick one of four options; the costs are a linear function of the features."""

    name = "assignment"

    def __init__(self):
        generator = torch.Generator().manual_seed(0)
        self.matrix = torch.randn(3, 4, generator=generator)
        self.data = {}
        for split, size in (("train", 64), ("val", 32), ("test", 32)):
            features = torch.randn(size, 3, generator=generator)
            self.data[split] = (features, features @ self.matrix)

    def load_data(self, split):
        return self.data[split]

    def solve(self, parameters, context=None):
        return torch.nn.functional.one_hot(parameters.argmin(-1), 4).float()

    def cost(self, decisions, parameters):
        return (decisions * parameters).sum(-1)


def test_base_class_methods_must_be_implemented():
    problem = Problem()
    for call in (lambda: problem.load_data("train"), lambda: problem.solve(torch.zeros(1, 2)),
                 lambda: problem.cost(torch.zeros(1, 2), torch.zeros(1, 2))):
        with pytest.raises(NotImplementedError):
            call()
    assert problem.context("train") is None
    decisions = torch.ones(2, 3)
    assert problem.recourse(decisions, decisions) is decisions


def test_named_and_custom_configurations():
    assert set(CONFIGS) == {"cosA", "cosB", "flatSNN", "flatMoE"}
    assert get_config("flatSNN").rank == 4
    custom = get_config({"epsilon": [0.3, 0.05], "subspace": "adaptive", "rank": 8})
    assert custom.epsilon == (0.3, 0.05) and custom.subspace == "adaptive"
    assert get_config("cosB", rank=4, max_subspace_dim=None).rank == 4
    with pytest.raises(KeyError):
        get_config("cosC")
    with pytest.raises(KeyError):
        get_config("cosB", learning_rate=0.1)


def test_learner_improves_a_subclass_from_realized_costs():
    problem = Assignment()
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 4)
    before = float(evaluate(problem, model).mean())
    result = run(problem, model, config="cosA", steps=40, batch_size=32, seed=0)
    assert result["test_cost"] < before
    assert result["stopped_at"] <= 40 and result["val_best"] is not None
    assert len(result["train_costs"]) == result["stopped_at"]


def test_optimizer_for_external_loops():
    model = torch.nn.Sequential(torch.nn.Linear(3, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    optimizer = Learner("flatSNN").optimizer(model, seed=0, horizon=10)
    params = dict(model.named_parameters())
    before = {k: v.detach().clone() for k, v in params.items()}
    features = torch.randn(8, 3)

    def closure(stacked):
        out = torch.func.vmap(lambda p: torch.func.functional_call(model, p, (features,)))(stacked)
        return (out ** 2).mean(dim=(1, 2))

    optimizer.step(closure)
    assert any(not torch.equal(before[k], v) for k, v in params.items())
