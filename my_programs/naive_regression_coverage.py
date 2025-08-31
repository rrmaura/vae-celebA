# Program to run naive regression 1000 times and calculate coverage
# This runs simple OLS regression without DDML to estimate causal effects

import numpy as np
import pandas as pd
import statsmodels.api as sm
import csv
import os
from CV_data_generation import randomize_data
from datetime import date

# Get today's date for filename
today = date.today()
d1 = today.strftime("%d_%m_%Y")

# Parameters
num_simulations = 1000
random_param = False  # Set to True if you want random DGP parameters
complicated = True  # Set to False for simple DGP

# Create CSV file to save results
name_csv = f"my_csv/{d1}_naive_regression_results.csv"
if not os.path.isfile(name_csv):
    with open(name_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed",
                "ATE_naive",
                "SE_naive",
                "CI_lower",
                "CI_upper",
                "ATE_with_male",
                "SE_with_male",
                "CI_male_lower",
                "CI_male_upper",
                "ATE_with_controls",
                "SE_with_controls",
                "CI_controls_lower",
                "CI_controls_upper",
                "ATE_truth",
                "naive_in_CI",
                "male_in_CI",
                "controls_in_CI",
            ]
        )

coverage_count_naive = 0
coverage_count_male = 0
coverage_count_controls = 0

for i in range(num_simulations):
    print(f"Simulation {i+1}/{num_simulations}")

    # Generate random seed for this simulation
    seed = np.random.randint(100000)
    np.random.seed(seed)

    # Generate new data for this simulation
    ATE_truth, male_on_job, male_on_master = randomize_data(
        seed, random_param=random_param, complicated=complicated
    )

    # Load the generated data
    data = pd.read_csv("my_csv/features_MSC_JOB.csv")

    Y = data["Job"]
    T = data["Master"]  # Treatment variable
    Male = data["Male"]
    Young = data["Young"]

    # 1. NAIVE REGRESSION: Just regress Y on T (ignoring confounders)
    X_naive = sm.add_constant(T)
    model_naive = sm.OLS(Y, X_naive).fit()
    ATE_naive = model_naive.params[1]  # Coefficient on Master
    SE_naive = model_naive.bse[1]
    CI_naive = model_naive.conf_int(alpha=0.05)[1]  # 95% CI for Master coefficient

    # 2. REGRESSION WITH MALE CONTROL: regress Y on T and Male
    X_male = data[["Master", "Male"]]
    X_male = sm.add_constant(X_male)
    model_male = sm.OLS(Y, X_male).fit()
    ATE_with_male = model_male.params[1]  # Coefficient on Master
    SE_with_male = model_male.bse[1]
    CI_male = model_male.conf_int(alpha=0.05)[1]

    # 3. REGRESSION WITH MORE CONTROLS: regress Y on T, Male, and Young
    X_controls = data[["Master", "Male", "Young"]]
    X_controls = sm.add_constant(X_controls)
    model_controls = sm.OLS(Y, X_controls).fit()
    ATE_with_controls = model_controls.params[1]  # Coefficient on Master
    SE_with_controls = model_controls.bse[1]
    CI_controls = model_controls.conf_int(alpha=0.05)[1]

    # Check if true ATE falls within confidence intervals
    naive_in_CI = int(CI_naive[0] <= ATE_truth <= CI_naive[1])
    male_in_CI = int(CI_male[0] <= ATE_truth <= CI_male[1])
    controls_in_CI = int(CI_controls[0] <= ATE_truth <= CI_controls[1])

    # Update coverage counters
    coverage_count_naive += naive_in_CI
    coverage_count_male += male_in_CI
    coverage_count_controls += controls_in_CI

    # Save results to CSV
    with open(name_csv, "a") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                seed,
                ATE_naive,
                SE_naive,
                CI_naive[0],
                CI_naive[1],
                ATE_with_male,
                SE_with_male,
                CI_male[0],
                CI_male[1],
                ATE_with_controls,
                SE_with_controls,
                CI_controls[0],
                CI_controls[1],
                ATE_truth,
                naive_in_CI,
                male_in_CI,
                controls_in_CI,
            ]
        )

# Calculate and print final coverage rates
coverage_naive = coverage_count_naive / num_simulations
coverage_male = coverage_count_male / num_simulations
coverage_controls = coverage_count_controls / num_simulations

print(f"\n=== COVERAGE RESULTS AFTER {num_simulations} SIMULATIONS ===")
print(f"Naive regression (no controls) coverage: {coverage_naive:.3f}")
print(f"Regression with Male control coverage: {coverage_male:.3f}")
print(f"Regression with Male + Young controls coverage: {coverage_controls:.3f}")
print(f"Expected coverage (95% CI): 0.950")
print(f"\nResults saved to: {name_csv}")
