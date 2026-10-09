"""Print the two-plan tradeoff and the equation (1) posterior."""

from lever.search import Draw, Metric, Search, posterior


def main():
    metric = Metric("comp", scale="log", sigma=1.0, rho=1.0)
    v = posterior(4.0, [2.0, 8.0], 0, metric)
    print(f"posterior V={v:.6f}")

    def expander(name, attempt):
        if name != "root":
            return None
        if attempt == 0:
            return Draw("plan_a", 1.0, metrics={"imp": 10.0})
        if attempt == 1:
            return Draw("plan_b", 5.0, metrics={"imp": 1.0})
        return None

    priors = {"comp": 4.0, "imp": 1.0}
    metrics = [
        Metric("comp", scale="log", sigma=1.0, rho=1.0),
        Metric("imp", scale="linear", sigma=1.0, rho=1.0),
    ]
    for lam in (0.0, 1.0):
        result = Search(metrics, lambdas={"imp": lam}, delta=1.2, budget=100.0).run(
            "root", priors, expander
        )
        chosen = ",".join(result.chosen)
        print(f"lambda={lam:.0f} chosen={chosen} C={result.C:.6f} solved={result.solved}")


if __name__ == "__main__":
    main()
