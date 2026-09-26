"""The interface a decision problem implements to be trained from realized costs."""


class Problem:
    """A decision problem: its data, a solver and the realized cost of a decision.

    Subclass it and implement load_data, solve and cost. Costs are minimized, so a
    problem that maximizes a value returns the negated value. A problem whose solver needs
    per-instance data besides the predicted parameters returns it from context; a problem
    with a recourse action repairs a decision in recourse and charges it in cost.
    """

    name = "problem"

    def load_data(self, split):
        """Return the features and the true parameters of a split (train, val or test)."""
        raise NotImplementedError

    def context(self, split):
        """Return per-instance solver data of a split aligned with load_data, or None."""
        return None

    def solve(self, parameters, context=None):
        """Return one decision per row of predicted parameters."""
        raise NotImplementedError

    def cost(self, decisions, parameters):
        """Return the realized cost of each decision under the true parameters."""
        raise NotImplementedError

    def recourse(self, decisions, parameters):
        """Return the decisions repaired under the true parameters, unchanged by default."""
        return decisions
