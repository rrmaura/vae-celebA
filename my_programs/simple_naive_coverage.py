# Simple program to run naive regression 1000 times and calculate coverage
# Just does Y ~ Master (no controls) to see bias and coverage

import numpy as np
import pandas as pd
import statsmodels.api as sm
import csv
import os
import matplotlib.pyplot as plt
from CV_data_generation import randomize_data

# Parameters
num_simulations = 1000
random_param = False  # Set to True if you want random DGP parameters
complicated = True  # Set to False for simple DGP

# Create CSV file to save results
name_csv = "my_csv/simple_naive_results.csv"
if not os.path.isfile(name_csv):
    with open(name_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "simulation",
                "ATE_estimate",
                "SE",
                "CI_lower",
                "CI_upper",
                "ATE_truth",
                "in_CI",
                "bias",
            ]
        )

results = []
coverage_count = 0

print("Running 1000 naive regressions...")

for i in range(num_simulations):
    if (i + 1) % 100 == 0:
        print(f"Completed {i+1}/{num_simulations} simulations")

    # Generate random seed for this simulation
    seed = np.random.randint(100000)

    # Generate new data for this simulation
    ATE_truth, _, _ = randomize_data(
        seed, random_param=random_param, complicated=complicated
    )

    # Load the generated data
    data = pd.read_csv("my_csv/features_MSC_JOB.csv")

    Y = data["Job"]  # Outcome: whether person got job
    T = data["Master"]  # Treatment: whether person has master's degree

    # NAIVE REGRESSION: Just regress Y on T (ignoring all confounders)
    X = sm.add_constant(T)
    model = sm.OLS(Y, X).fit()

    ATE_estimate = model.params[1]  # Coefficient on Master
    SE = model.bse[1]  # Standard error
    CI = model.conf_int(alpha=0.05)[1]  # 95% confidence interval

    # Check if true ATE falls within confidence interval
    in_CI = int(CI[0] <= ATE_truth <= CI[1])
    bias = ATE_estimate - ATE_truth

    coverage_count += in_CI

    # Store results
    results.append(
        {
            "simulation": i + 1,
            "ATE_estimate": ATE_estimate,
            "SE": SE,
            "CI_lower": CI[0],
            "CI_upper": CI[1],
            "ATE_truth": ATE_truth,
            "in_CI": in_CI,
            "bias": bias,
        }
    )

    # Save to CSV
    with open(name_csv, "a") as f:
        writer = csv.writer(f)
        writer.writerow([i + 1, ATE_estimate, SE, CI[0], CI[1], ATE_truth, in_CI, bias])

# Calculate final statistics
coverage_rate = coverage_count / num_simulations
estimates = [r["ATE_estimate"] for r in results]
biases = [r["bias"] for r in results]
true_ates = [r["ATE_truth"] for r in results]

print(f"\n=== RESULTS AFTER {num_simulations} SIMULATIONS ===")
print(f"Coverage rate: {coverage_rate:.3f} (expected: 0.950)")
print(f"Average ATE estimate: {np.mean(estimates):.4f}")
print(f"Average true ATE: {np.mean(true_ates):.4f}")
print(f"Average bias: {np.mean(biases):.4f}")
print(f"Standard deviation of estimates: {np.std(estimates):.4f}")
print(f"Mean absolute bias: {np.mean(np.abs(biases)):.4f}")
print(f"\nResults saved to: {name_csv}")

# Quick interpretation
if coverage_rate < 0.90:
    print("\n⚠️  Coverage is much lower than 95% - this suggests significant bias!")
elif coverage_rate < 0.93:
    print("\n⚠️  Coverage is somewhat low - there may be some bias.")
else:
    print("\n✅ Coverage looks reasonable.")

if abs(np.mean(biases)) > 0.01:
    print(f"📊 Average bias of {np.mean(biases):.4f} suggests omitted variable bias.")

# Create and save plots
print("\nCreating plots...")

# 1. Histogram of ATE estimates vs true ATE
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.hist(estimates, bins=50, alpha=0.7, density=True, label="ATE Estimates")
plt.axvline(
    np.mean(true_ates),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"True ATE (avg: {np.mean(true_ates):.4f})",
)
plt.axvline(
    np.mean(estimates),
    color="blue",
    linestyle="--",
    linewidth=2,
    label=f"Mean Estimate: {np.mean(estimates):.4f}",
)
plt.xlabel("ATE Estimate")
plt.ylabel("Density")
plt.title("Distribution of ATE Estimates")
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Histogram of bias
plt.subplot(2, 2, 2)
plt.hist(biases, bins=50, alpha=0.7, color="orange")
plt.axvline(0, color="red", linestyle="--", linewidth=2, label="No Bias")
plt.axvline(
    np.mean(biases),
    color="blue",
    linestyle="--",
    linewidth=2,
    label=f"Mean Bias: {np.mean(biases):.4f}",
)
plt.xlabel("Bias (Estimate - True ATE)")
plt.ylabel("Frequency")
plt.title("Distribution of Bias")
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Coverage indicator over simulations
plt.subplot(2, 2, 3)
coverage_indicators = [r["in_CI"] for r in results]
running_coverage = np.cumsum(coverage_indicators) / np.arange(
    1, len(coverage_indicators) + 1
)
plt.plot(running_coverage, alpha=0.8)
plt.axhline(0.95, color="red", linestyle="--", label="Target Coverage (95%)")
plt.axhline(
    coverage_rate,
    color="blue",
    linestyle="--",
    label=f"Final Coverage: {coverage_rate:.3f}",
)
plt.xlabel("Simulation Number")
plt.ylabel("Running Coverage Rate")
plt.title("Coverage Rate Over Simulations")
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Scatter plot: True ATE vs Estimated ATE
plt.subplot(2, 2, 4)
plt.scatter(true_ates, estimates, alpha=0.6, s=20)
plt.plot(
    [min(true_ates), max(true_ates)],
    [min(true_ates), max(true_ates)],
    "r--",
    label="Perfect Estimation",
)
plt.xlabel("True ATE")
plt.ylabel("Estimated ATE")
plt.title("True vs Estimated ATE")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("my_csv/naive_regression_results.png", dpi=300, bbox_inches="tight")
plt.savefig("my_csv/naive_regression_results.pdf", bbox_inches="tight")
print("📊 Plots saved as 'my_csv/naive_regression_results.png' and '.pdf'")

# 5. Create a separate plot for standardized estimates (to check normality)
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
# Standardized estimates: (estimate - true) / SE
standardized = [
    (estimates[i] - true_ates[i]) / results[i]["SE"] for i in range(len(results))
]
plt.hist(standardized, bins=50, alpha=0.7, density=True, label="Standardized Estimates")

# Overlay standard normal for comparison
x_norm = np.linspace(-4, 4, 100)
y_norm = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_norm**2)
plt.plot(x_norm, y_norm, "r-", linewidth=2, label="Standard Normal")
plt.xlabel("(Estimate - True ATE) / SE")
plt.ylabel("Density")
plt.title("Standardized Estimates vs Standard Normal")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Q-Q plot
from scipy import stats

stats.probplot(standardized, dist="norm", plot=plt)
plt.title("Q-Q Plot: Standardized Estimates vs Normal")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("my_csv/naive_regression_normality.png", dpi=300, bbox_inches="tight")
print("📊 Normality check plots saved as 'my_csv/naive_regression_normality.png'")

plt.show()
