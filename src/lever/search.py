"""AND/OR proof search from LEVER (arXiv 2610.11862, section 3).

OR nodes are goals. AND nodes are decompositions. Each goal also carries a prior
AND node that is never expanded: it holds the current estimate of finishing the
goal with a fresh decomposition. Selection descends the argmin child at an OR
node and every child at an AND node. Landing on a prior expands that goal. A
goal that already has a decomposition is redrawn only when the incumbent
objective is at least ``delta`` times the prior, the paper's "beat the incumbent
by a factor delta".

Values sum at AND nodes and take the min at OR nodes. An unsolved goal with no
finite decomposition costs infinity. Vcomp, the remaining computational cost,
adds nothing for a realized edge and the prior estimate for an open goal.
Other metrics add their edge costs. The objective is

    C(n) = Vcomp(n) + sum_m lambda_m * V_m(n)

Equation (1) updates a goal's estimate from the backed-up costs of its
decompositions. A failed attempt adds one sigma and is not a cost observation.
Log-scale metrics (computational cost, length) use the geometric form. Linear
metrics (impurity, counted in references) use the same average without logs.

A shared subgoal is charged to the first decomposition that introduced it.
Later listings of the same name reuse that node and do not add its cost again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


INF = math.inf


@dataclass(frozen=True)
class Metric:
    """One cost that composes over the AND/OR algebra.

    ``scale`` is ``"log"`` or ``"linear"``. ``sigma`` is the spread of repeated
    draws, ``rho`` the value function's error. Both are in log space for a log
    metric and in the metric's own units for a linear one. ``n = sigma^2 / rho^2``
    is the number of draws the prior is worth.
    """

    name: str
    scale: str = "log"
    sigma: float = 1.0
    rho: float = 1.0

    def __post_init__(self):
        if self.scale not in ("log", "linear"):
            raise ValueError(f"scale must be 'log' or 'linear', got {self.scale!r}")
        if self.rho == 0:
            raise ValueError("rho must be nonzero")
        if self.scale == "log" and self.sigma < 0:
            raise ValueError("sigma must be non-negative")


@dataclass(frozen=True)
class Subgoal:
    name: str
    priors: dict


@dataclass(frozen=True)
class Draw:
    """One successful expansion.

    ``comp_cost`` is the computational cost of this expansion. It is sunk spend,
    the observation for the comp posterior, and the edge cost of the additive
    comp metric. ``metrics`` holds edge costs of the other metrics on this
    decomposition (impurity of the proof text, length, and so on). An empty
    ``subgoals`` tuple is a complete proof of the goal.
    """

    name: str
    comp_cost: float
    metrics: dict = field(default_factory=dict)
    subgoals: tuple = ()

    def __post_init__(self):
        if self.comp_cost <= 0:
            raise ValueError("comp_cost must be positive")
        object.__setattr__(self, "subgoals", tuple(self.subgoals))


class Disproof:
    """The expander reports the goal false. Its value becomes infinity."""


def posterior(v0, observations, failures, metric):
    """Equation (1), plus the failed-attempt rule.

    ``n = sigma^2 / rho^2``. For a log metric,

        mu = (n log V0 + sum log v_i) / (n + k),   V = exp(mu)

    and each failure adds ``sigma`` in log space, which multiplies the estimate
    by ``e^sigma``. A linear metric uses the same average without logarithms,
    and each failure adds ``sigma``. Failures are not cost observations.
    """
    observations = list(observations)
    failures = int(failures)
    n = (metric.sigma / metric.rho) ** 2
    k = len(observations)
    if metric.scale == "log":
        if v0 <= 0:
            raise ValueError("log-scale prior must be positive")
        for v in observations:
            if v <= 0:
                raise ValueError("log-scale observation must be positive")
        if k == 0:
            mu = math.log(v0)
        else:
            mu = (n * math.log(v0) + sum(math.log(v) for v in observations)) / (n + k)
        mu += failures * metric.sigma
        return math.exp(mu)
    if k == 0:
        mu = v0
    else:
        mu = (n * v0 + sum(observations)) / (n + k)
    return mu + failures * metric.sigma


def redraw_justified(incumbent, prior, delta):
    """True when a fresh draw must be taken.

    The paper requires the redraw to beat the incumbent by a factor ``delta``.
    That is ``incumbent / prior >= delta`` when both are finite and the prior
    is positive. An infinite incumbent loses to any finite prior. A non-positive
    prior beats any larger incumbent.
    """
    if delta < 1:
        raise ValueError("delta must be at least 1")
    if math.isinf(incumbent):
        return not math.isinf(prior)
    if math.isinf(prior):
        return False
    if prior <= 0:
        return incumbent > prior
    return incumbent / prior >= delta


@dataclass
class Result:
    solved: bool
    spent: float
    chosen: tuple
    solution_cost: float
    C: float
    root_additive: float


class _Goal:
    def __init__(self, name, priors):
        self.name = name
        self.v0 = dict(priors)
        self.decomps = []
        self.failures = 0
        self.impossible = False
        self.attempts = 0


class _Decomp:
    def __init__(self, name, edge, subgoals, owner_of):
        self.name = name
        self.edge = dict(edge)
        self.subgoals = list(subgoals)
        self.owner_of = set(owner_of)


class Search:
    """Run Algorithm 1 with an injected expander.

    ``expander(goal_name, attempt)`` returns a :class:`Draw`, :class:`Disproof`,
    or ``None``. ``None`` is a failed attempt: no cost observation, estimate
    raised by one sigma. ``attempt`` counts previous expansions of that goal,
    starting at 0.

    ``metrics`` must include one named ``comp``. ``lambdas`` prices the other
    metrics against remaining compute. ``budget`` is the computational spend
    cap. A proof counts as solved only when a complete solution subgraph exists
    and the spend is at most the budget.
    """

    def __init__(self, metrics, lambdas=None, delta=1.0, budget=INF, max_steps=1000):
        self.metrics = {m.name: m for m in metrics}
        if "comp" not in self.metrics:
            raise ValueError("metrics must include 'comp'")
        if self.metrics["comp"].scale != "log":
            raise ValueError("comp is a log-scale metric in the paper")
        self.lambdas = dict(lambdas or {})
        unknown = set(self.lambdas) - set(self.metrics)
        if unknown:
            raise ValueError(f"lambda for unknown metric: {sorted(unknown)}")
        if "comp" in self.lambdas:
            raise ValueError("comp is not priced by lambda; it is Vcomp")
        self.delta = float(delta)
        self.budget = float(budget)
        self.max_steps = int(max_steps)
        self.goals = {}
        self.owners = {}
        self.spent = 0.0
        self.steps = 0

    def run(self, root, priors, expander):
        self.goals = {root: _Goal(root, self._full_priors(priors))}
        self.owners = {}
        self.spent = 0.0
        self.steps = 0
        while self.spent < self.budget and self.steps < self.max_steps:
            targets = self._select(self.goals[root], set())
            if not targets:
                break
            progressed = False
            for goal in targets:
                if self.spent >= self.budget or self.steps >= self.max_steps:
                    break
                self._expand(goal, expander)
                progressed = True
            if not progressed:
                break
        return self._result(root)

    def _full_priors(self, priors):
        missing = set(self.metrics) - set(priors)
        if missing:
            raise ValueError(f"missing priors for {sorted(missing)}")
        return {name: float(priors[name]) for name in self.metrics}

    def _expand(self, goal, expander):
        attempt = goal.attempts
        goal.attempts += 1
        self.steps += 1
        drawn = expander(goal.name, attempt)
        if drawn is None:
            goal.failures += 1
            return
        if isinstance(drawn, Disproof):
            goal.impossible = True
            return
        if not isinstance(drawn, Draw):
            raise TypeError(f"expander returned {type(drawn).__name__}, want Draw, Disproof, or None")
        edge = {"comp": float(drawn.comp_cost)}
        for name, metric in self.metrics.items():
            if name == "comp":
                continue
            if name not in drawn.metrics:
                raise ValueError(f"draw {drawn.name!r} is missing metric {name!r}")
            edge[name] = float(drawn.metrics[name])
        owned = []
        attached = []
        for sub in drawn.subgoals:
            attached.append(sub.name)
            if sub.name not in self.goals:
                self.goals[sub.name] = _Goal(sub.name, self._full_priors(sub.priors))
                owned.append(sub.name)
                self.owners[sub.name] = drawn.name
            elif sub.name not in self.owners:
                self.owners[sub.name] = drawn.name
                owned.append(sub.name)
        self.spent += float(drawn.comp_cost)
        goal.decomps.append(_Decomp(drawn.name, edge, attached, owned))

    def _observations(self, goal, metric_name):
        # A disproved branch is not a cost sample. Dropping it keeps the prior
        # finite, so selection can redraw instead of treating the goal as hopeless.
        costs = []
        for decomp in goal.decomps:
            value = self._decomp_additive(decomp, metric_name, set())
            if not math.isinf(value):
                costs.append(value)
        return costs

    def _estimate(self, goal, metric_name):
        if goal.impossible:
            return INF
        metric = self.metrics[metric_name]
        return posterior(
            goal.v0[metric_name],
            self._observations(goal, metric_name),
            goal.failures,
            metric,
        )

    def _decomp_additive(self, decomp, metric_name, seen):
        total = decomp.edge[metric_name]
        for name in decomp.owner_of:
            if name in seen:
                continue
            seen = set(seen)
            seen.add(name)
            child = self._goal_additive(self.goals[name], metric_name, seen)
            if math.isinf(child):
                return INF
            total += child
        return total

    def _goal_additive(self, goal, metric_name, seen):
        if goal.impossible and not goal.decomps:
            return INF
        options = []
        if not goal.impossible:
            options.append(self._estimate(goal, metric_name))
        for decomp in goal.decomps:
            options.append(self._decomp_additive(decomp, metric_name, seen))
        if not options:
            return INF
        return min(options)

    def _decomp_vcomp(self, decomp, seen):
        total = 0.0
        for name in decomp.owner_of:
            if name in seen:
                continue
            seen = set(seen)
            seen.add(name)
            child = self._goal_vcomp(self.goals[name], seen)
            if math.isinf(child):
                return INF
            total += child
        return total

    def _goal_vcomp(self, goal, seen):
        if goal.impossible and not goal.decomps:
            return INF
        options = []
        if not goal.impossible:
            options.append(self._estimate(goal, "comp"))
        for decomp in goal.decomps:
            options.append(self._decomp_vcomp(decomp, seen))
        if not options:
            return INF
        return min(options)

    def _quality(self, kind, node, seen):
        total = 0.0
        for name, lam in self.lambdas.items():
            if lam == 0:
                continue
            if kind == "goal":
                value = self._goal_additive(node, name, seen)
            else:
                value = self._decomp_additive(node, name, seen)
            if math.isinf(value):
                return INF
            total += lam * value
        return total

    def _C_goal_prior(self, goal):
        if goal.impossible:
            return INF
        return self._estimate(goal, "comp") + self._quality("goal", goal, set())

    def _C_decomp(self, decomp):
        remaining = self._decomp_vcomp(decomp, set())
        if math.isinf(remaining):
            return INF
        extra = self._quality("decomp", decomp, set())
        if math.isinf(extra):
            return INF
        return remaining + extra

    def _select(self, goal, visiting):
        if goal.name in visiting or goal.impossible:
            return []
        options = []
        for decomp in goal.decomps:
            cost = self._C_decomp(decomp)
            if not math.isinf(cost):
                options.append((cost, decomp))
        prior = self._C_goal_prior(goal)
        if not math.isinf(prior):
            options.append((prior, None))
        if not options:
            return []
        _, best = min(options, key=lambda item: (item[0], "" if item[1] is None else item[1].name))
        if best is None:
            if not goal.decomps:
                return [goal]
            realized = [cost for cost, node in options if node is not None]
            incumbent = min(realized) if realized else INF
            if redraw_justified(incumbent, prior, self.delta):
                return [goal]
            realized = [(cost, node) for cost, node in options if node is not None]
            if not realized:
                return []
            _, best = min(realized, key=lambda item: (item[0], item[1].name))
        visiting = set(visiting)
        visiting.add(goal.name)
        selected = []
        for name in best.subgoals:
            selected.extend(self._select(self.goals[name], visiting))
        return selected

    def _chosen(self, goal, visiting):
        """Cheapest complete solution subgraph under C, then additive comp cost.

        Returns ``(C, additive, names)`` or None if this goal has no complete
        finite solution. The prior node is not a solution.
        """
        if goal.name in visiting:
            return None
        best = None
        visiting = set(visiting)
        visiting.add(goal.name)
        for decomp in goal.decomps:
            additive = decomp.edge["comp"]
            names = [decomp.name]
            blocked = False
            child_C_extra = 0.0
            for name in decomp.owner_of:
                child = self._chosen(self.goals[name], visiting)
                if child is None:
                    blocked = True
                    break
                additive += child[1]
                names.extend(child[2])
                child_C_extra += child[0]
            if blocked:
                continue
            # A complete decomposition has no remaining compute. Quality is the
            # backed-up metric cost, which _C_decomp already computes when every
            # owned subgoal is solved (its Vcomp is 0).
            cost = self._C_decomp(decomp)
            if math.isinf(cost):
                continue
            # Guard: an open owned subgoal means this decomposition is not a proof.
            if any(self._chosen(self.goals[n], visiting) is None for n in decomp.owner_of):
                continue
            cand = (cost, additive, tuple(names))
            if best is None or cand < best:
                best = cand
        return best

    def _result(self, root):
        goal = self.goals[root]
        chosen = self._chosen(goal, set())
        additive = self._goal_additive(goal, "comp", set())
        if chosen is None:
            return Result(False, self.spent, (), INF, self._C_goal_prior(goal), additive)
        cost, solution_cost, names = chosen
        solved = solution_cost < INF and self.spent <= self.budget
        return Result(solved, self.spent, names, solution_cost, cost, additive)
