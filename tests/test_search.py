"""Tests for the LEVER search rules in arXiv 2610.11862, section 3."""

import math

import pytest

from lever.search import Disproof, Draw, Metric, Search, Subgoal, posterior, redraw_justified


COMP = Metric("comp", scale="log", sigma=1.0, rho=1.0)
IMP = Metric("imp", scale="linear", sigma=1.0, rho=1.0)


def _search(**kwargs):
    metrics = kwargs.pop("metrics", [COMP])
    return Search(metrics, **kwargs)


def test_posterior_matches_equation_1():
    # V0 = 4, costs 2 and 8, sigma = rho = 1, so n = 1.
    # mu = (log 4 + log 2 + log 8) / 3 = log(64) / 3 = log 4.
    metric = Metric("comp", scale="log", sigma=1.0, rho=1.0)
    got = posterior(4.0, [2.0, 8.0], 0, metric)
    assert got == pytest.approx(4.0)


def test_failed_attempt_multiplies_log_estimate_by_e_sigma():
    metric = Metric("comp", scale="log", sigma=1.0, rho=1.0)
    got = posterior(4.0, [], 1, metric)
    assert got == pytest.approx(4.0 * math.e)


def test_linear_failure_adds_sigma():
    metric = Metric("imp", scale="linear", sigma=1.0, rho=1.0)
    got = posterior(4.0, [], 1, metric)
    assert got == pytest.approx(5.0)
    # An observation is averaged, not logged.
    got = posterior(4.0, [10.0], 0, metric)
    assert got == pytest.approx((4.0 + 10.0) / 2)


def test_redraw_margin():
    # incumbent / prior >= delta is the factor-delta test.
    assert redraw_justified(10.0, 9.0, 1.2) is False
    assert redraw_justified(10.0, 8.0, 1.2) is True
    assert redraw_justified(math.inf, 1.0, 1.2) is True
    assert redraw_justified(1.0, math.inf, 1.2) is False


def test_exact_priors_find_the_cheaper_proof():
    def expander(name, attempt):
        if name == "root" and attempt == 0:
            return Draw("via_left", 1.0, subgoals=(Subgoal("left", {"comp": 10.0}),))
        if name == "root" and attempt == 1:
            return Draw("via_right", 1.0, subgoals=(Subgoal("right", {"comp": 2.0}),))
        if name == "right" and attempt == 0:
            return Draw("right_proof", 2.0)
        if name == "left" and attempt == 0:
            return Draw("left_proof", 10.0)
        return None

    # delta=1: the prior 3 beats the realized 11, so the redraw is taken.
    result = _search(delta=1.0, budget=100.0).run("root", {"comp": 3.0}, expander)
    assert result.solved
    assert result.chosen == ("via_right", "right_proof")
    assert result.solution_cost == pytest.approx(3.0)
    assert result.root_additive == pytest.approx(3.0)
    assert "via_left" not in result.chosen


def test_lambda_trades_find_cost_against_impurity():
    def expander(name, attempt):
        if name != "root":
            return None
        if attempt == 0:
            return Draw("plan_a", 1.0, metrics={"imp": 10.0})
        if attempt == 1:
            return Draw("plan_b", 5.0, metrics={"imp": 1.0})
        return None

    priors = {"comp": 4.0, "imp": 1.0}
    cheap = Search([COMP, IMP], lambdas={"imp": 0.0}, delta=1.2, budget=100.0)
    a = cheap.run("root", priors, expander)
    assert a.solved
    assert a.chosen == ("plan_a",)
    assert a.C == pytest.approx(0.0)

    pure = Search([COMP, IMP], lambdas={"imp": 1.0}, delta=1.2, budget=100.0)
    b = pure.run("root", priors, expander)
    assert b.solved
    assert b.chosen == ("plan_b",)
    assert b.C == pytest.approx(1.0)


def test_proof_over_budget_is_not_solved():
    def expander(name, attempt):
        if attempt == 0:
            return Draw("only", 10.0)
        return None

    result = _search(budget=5.0).run("root", {"comp": 1.0}, expander)
    assert result.solved is False
    assert result.spent == pytest.approx(10.0)
    assert result.solution_cost == pytest.approx(10.0)


def test_infinite_branch_is_not_chosen():
    def expander(name, attempt):
        if name == "root" and attempt == 0:
            return Draw("bad", 1.0, subgoals=(Subgoal("dead", {"comp": 1.0}),))
        if name == "dead" and attempt == 0:
            return Disproof()
        if name == "root" and attempt == 1:
            return Draw("good", 1.0)
        return None

    result = _search(delta=1.0, budget=100.0).run("root", {"comp": 3.0}, expander)
    assert result.solved
    assert result.chosen == ("good",)
    assert "bad" not in result.chosen
    assert result.solution_cost == pytest.approx(1.0)


def test_shared_subgoal_is_charged_once():
    def expander(name, attempt):
        if name == "root" and attempt == 0:
            return Draw("owner", 1.0, subgoals=(Subgoal("shared", {"comp": 4.0}),))
        if name == "root" and attempt == 1:
            return Draw("borrower", 1.0, subgoals=(Subgoal("shared", {"comp": 4.0}),))
        if name == "shared" and attempt == 0:
            return Draw("shared_proof", 4.0)
        return None

    search = _search(delta=1.0, budget=100.0)
    result = search.run("root", {"comp": 2.0}, expander)
    assert result.solved
    # One node, expanded once. The borrower does not add the shared cost.
    assert search.goals["shared"].attempts == 1
    owner = search.goals["root"].decomps[0]
    borrower = search.goals["root"].decomps[1]
    assert search._decomp_additive(owner, "comp", set()) == pytest.approx(5.0)
    assert search._decomp_additive(borrower, "comp", set()) == pytest.approx(1.0)
