# Data Science Internship Projects

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Notebooks-Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

## Introduction

This repository documents two applied machine learning tasks completed during a Data Science Internship: classifying Iris flowers from physical measurements and estimating California census block-group median house values. Each project is implemented as a Jupyter notebook using its accompanying CSV file. The notebooks include exploratory analysis, preprocessing, model training, and evaluation; this README summarizes the methods and the results saved in those notebooks.

> **Reproducibility note:** Reported scores are from the notebooks' saved runs. They describe the specific train/test splits and should not be interpreted as guarantees of performance on future or independently collected data.

## Internship Project Overview

| Task | Problem type | Prediction target | Main comparison |
|---|---|---|---|
| Iris classification | Multiclass classification | Flower species | Logistic Regression vs. Decision Tree |
| House price prediction | Regression | Census-block-group median house value | OLS, Ridge, and Lasso with original vs. engineered features |

## Project Objectives

- Inspect real tabular datasets and summarize data quality and distributions.
- Apply preprocessing suitable for numeric and categorical predictors.
- Compare appropriate supervised learning models on held-out data.
- Use class-level and regression metrics to interpret model performance.
- Communicate the scope and limitations of each experiment.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── Task 1/
│   ├── Iris_classificaon.ipynb
│   └── iris.csv
└── Task 2/
    ├── House_prediction.ipynb
    └── housing.csv
```

The Task 1 notebook's current filename is **`Iris_classificaon.ipynb`** (spelled as it appears in the repository). Both notebooks load their local CSV; the project does not download data or overwrite the source files. A dataset citation or upstream source URL is not documented in the project files.

## Task 1: Iris Classification

Notebook: [`Task 1/Iris_classificaon.ipynb`](Task%201/Iris_classificaon.ipynb)  
Dataset: [`Task 1/iris.csv`](Task%201/iris.csv)

### Problem Statement

Predict one of three Iris species—`setosa`, `versicolor`, or `virginica`—from four measurements. This is a multiclass classification problem: the model assigns one species label to each flower based on its sepal and petal lengths and widths.

### Dataset Overview

| Property | Details |
|---|---|
| Observations | 150 |
| Columns | 5: four measurements and the species label |
| Input features | `sepal_length`, `sepal_width`, `petal_length`, `petal_width` |
| Target | `species` |
| Class distribution | 50 `setosa`, 50 `versicolor`, 50 `virginica` |
| Missing values | None in the supplied CSV |

The notebook loads the repository's CSV with pandas; it does not use scikit-learn's built-in Iris dataset. The dataset source is not cited in the project files.

### Data Preprocessing

- The notebook checks column types, missing values, summary statistics, and duplicate rows.
- It detects **one duplicate row**, but does not remove it. The modeling data therefore retains all 150 rows; this should be considered when interpreting the holdout results.
- Rows without a target label would be excluded, although the supplied dataset has no missing labels.
- A stratified 80/20 split creates 120 training and 30 test rows, preserving the equal class proportions. The split uses a fixed random seed (`42`).
- A median imputer and standard scaler are fitted on training features only. Logistic Regression uses the imputed and standardized features; the Decision Tree uses imputed but unscaled features, since trees do not require standardization.

### Exploratory Data Analysis

The notebook includes a class-count plot, feature boxplots grouped by species, and a correlation heatmap for the four measurements. The classes are balanced. The boxplots and correlation plot help inspect how measurements vary and overlap by species; the notebook does not report statistical significance tests or use EDA alone as evidence of test performance.

### Machine Learning Approach

- **Logistic Regression** is used as a linear classification baseline. Standardization puts the measurements on comparable scales for model fitting.
- **Decision Tree** provides a rule-based model that can capture nonlinear splits and is trained without scaling.
- Both models are evaluated on the same held-out observations to make the comparison consistent.
- The notebook also plots two-feature decision boundaries as a visualization; these plots are illustrative and are distinct from the evaluation of the models trained with all four features.

### Model Evaluation and Actual Results

The saved notebook outputs report the following results on the 30-row test set:

| Model | Accuracy | Weighted precision | Weighted recall | Weighted F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.933 | 0.933 | 0.933 | 0.933 |
| Decision Tree | 0.933 | 0.933 | 0.933 | 0.933 |

Both confusion matrices are identical:

| Actual \ Predicted | setosa | versicolor | virginica |
|---|---:|---:|---:|
| setosa | 10 | 0 | 0 |
| versicolor | 0 | 9 | 1 |
| virginica | 0 | 1 | 9 |

Each model correctly classifies all 10 `setosa` examples and makes one error in each of the other two classes. The models tie on the reported metrics; this experiment does not show that one is generally superior.

### Key Findings

- The balanced target classes make overall accuracy informative for this split, while class-level reports show which species are confused.
- `versicolor` and `virginica` account for the observed test errors; both models make the same two mistakes.
- A duplicate is present and not removed, and performance is measured on one split only. Repeated or nested validation and duplicate handling would provide a stronger estimate of generalization.

### Real-World Applications

The workflow illustrates how measurement-based classification can support species identification and biological data exploration. The notebook is an educational experiment on a small, well-structured dataset—not a validated field identification or ecological decision system.

## Task 2: House Price Prediction

Notebook: [`Task 2/House_prediction.ipynb`](Task%202/House_prediction.ipynb)  
Dataset: [`Task 2/housing.csv`](Task%202/housing.csv)

### Problem Statement

Estimate `median_house_value` for a California census block group from its location, housing, population, income, and ocean-proximity attributes. This is a regression task. The target represents an area-level median, not the sale price of an individual property.

### Dataset Overview

| Property | Details |
|---|---|
| Observations | 20,640 |
| Columns | 10 |
| Numeric predictors | `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income` |
| Categorical predictor | `ocean_proximity` |
| Target | `median_house_value` |
| Missing data | 207 missing values in `total_bedrooms` |
| Target range in CSV | $14,999 to $500,001 |

The CSV and notebook identify the data as California housing data, but do not document an upstream dataset citation. The target is capped at $500,001, which limits interpretation near the upper end.

### Data Preprocessing

- The notebook checks data types, summary statistics, missing values, and duplicate records. The supplied CSV has no duplicate records.
- It creates a separate modeling copy and reserves a random 20% holdout (16,512 training rows and 4,128 test rows). The original dataframe is left unchanged.
- Three row-wise features are engineered: `rooms_per_household`, `bedrooms_per_room`, and `population_per_household`.
- Variance Inflation Factor (VIF) is calculated from training numeric features. With a cutoff of 10, `total_bedrooms` and `households` are removed from the candidate feature sets due to multicollinearity.
- Numeric columns are median-imputed and standardized; `ocean_proximity` is most-frequent-imputed and one-hot encoded. These transformations are inside scikit-learn pipelines, so they are fitted within each training fold rather than on the holdout test set.
- The VIF screening uses training data; the reserved holdout is not used to fit preprocessing or select Ridge/Lasso regularization strengths.

### Exploratory Data Analysis

The notebook visualizes the target distribution and boxplot, ocean-proximity counts, a longitude/latitude scatter plot colored by target value, and a numeric correlation heatmap. The saved output reports a mean target value of **$206,856** and a median of **$179,700**. `median_income` has the largest positive pairwise numeric correlation with the target in the notebook's output (**0.688**). The location plot provides spatial context, while the notebook notes that the linear models do not explicitly represent geographic neighborhoods.

### Machine Learning Approach

Two feature sets are compared: an original-feature baseline after VIF screening, and a set that adds the three household ratios. Each feature set is evaluated with:

- **Ordinary Least Squares (OLS)** as an unregularized linear baseline.
- **Ridge regression**, which shrinks coefficients to reduce sensitivity to correlated predictors.
- **Lasso regression**, which penalizes coefficient magnitude and can set coefficients to zero.

Ridge and Lasso hyperparameters are selected by `GridSearchCV` over six alpha values (`0.01` through `1,000`) using shuffled five-fold cross-validation on the training data. The search optimizes negative RMSE. The final model comparison uses the separate holdout set.

### Model Evaluation and Actual Results

The notebook reports holdout Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (`R²`). RMSE and MAE are expressed in dollars; lower is better for both error metrics. Higher R² is better.

| Feature set | Model | RMSE (USD) | MAE (USD) | R² |
|---|---|---:|---:|---:|
| Engineered | OLS | 70,308.834 | 50,594.977 | 0.623 |
| Engineered | Ridge | 70,316.153 | 50,601.570 | 0.623 |
| Engineered | Lasso | 70,320.125 | 50,604.100 | 0.623 |
| Original | OLS | 71,466.832 | 52,089.869 | 0.610 |
| Original | Ridge | 71,475.021 | 52,097.531 | 0.610 |
| Original | Lasso | 71,479.455 | 52,099.129 | 0.610 |

**Best result in this run:** OLS with engineered features, with **$70,308.834 RMSE**, **$50,594.977 MAE**, and **0.623 R²**. The search selected Ridge `alpha = 1` and Lasso `alpha = 10` for the engineered feature set. Residual and actual-versus-predicted plots are included in the notebook; the remaining error is substantial, and this result is not a property-level valuation.

### Key Findings

- Adding the engineered ratios improved the reported holdout metrics for all three linear model types compared with the VIF-screened original feature set in this split.
- OLS, Ridge, and Lasso scores within each feature set are close; the scores do not show a large advantage for regularization here.
- Median income and geographic coordinates have some of the largest continuous-feature coefficients in the selected OLS model. Coefficients describe fitted associations, not causal effects.
- The $500,001 target cap, spatial structure, and a single holdout split constrain how broadly these findings can be applied.

### Real-World Applications

Area-level estimates can support exploratory housing-market analysis, geographic comparisons, and planning discussions. This notebook does **not** produce current individual-home valuations or a deployable pricing service; such use would require current, appropriately licensed data, stronger spatial/nonlinear modeling, repeated validation, and monitoring.

## Technologies and Libraries Used

| Tool or library | Use in this project |
|---|---|
| Python | Notebook execution and modeling |
| JupyterLab / Jupyter Notebook | Interactive project notebooks |
| pandas | CSV loading and tabular analysis |
| NumPy | Numeric operations and feature engineering |
| Matplotlib and Seaborn | EDA, evaluation, and model visualizations |
| scikit-learn | Preprocessing, splitting, model fitting, cross-validation, and metrics |

The dependency ranges are listed in [`requirements.txt`](requirements.txt).

## Installation and Setup

Create a virtual environment from the repository root. The following commands are for Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, use the Python executable in `.venv\Scripts\python.exe` directly to install or launch tools. The dependency manifest specifies compatible version ranges rather than exact lockfile versions.

## How to Run the Project

From the activated environment and repository root:

```powershell
python -m jupyter lab
```

Open either notebook, select the available Python kernel, and run cells from top to bottom:

1. `Task 1/Iris_classificaon.ipynb`
2. `Task 2/House_prediction.ipynb`

Each notebook searches for its corresponding CSV in the task folder or project root. Keep the CSV beside its notebook for the standard repository layout. To reproduce the saved evaluation, use the notebook's existing random seed and workflow; dependency versions and numerical library differences may cause small variations.

## Key Learnings and Skills Gained

- Performing structured EDA and checking missing values, duplicates, class balance, distributions, and correlations.
- Separating model fitting from evaluation with holdout sets and stratified splitting for classification.
- Fitting imputation and scaling on training data to avoid leakage into evaluation.
- Encoding categorical predictors and using pipelines with cross-validation.
- Comparing classification models with class-level metrics and confusion matrices.
- Comparing regression models with dollar-based error measures and R².
- Engineering interpretable ratios and screening multicollinearity with VIF.
- Interpreting model coefficients cautiously and documenting data and evaluation limitations.

## Future Improvements

- For Iris, remove or explicitly investigate the duplicate record, and evaluate model stability with repeated stratified cross-validation.
- For housing, compare nonlinear and spatially aware models, while retaining a final untouched test set.
- Report cross-validation variability and prediction intervals, and investigate errors by geographic region and target-value range.
- Record dataset provenance, license, and exact environment versions for reproducibility.
- If intended for practical use, validate against recent, representative data and assess fairness, drift, and operational requirements before deployment.

## Conclusion

Together, the notebooks demonstrate two different supervised learning workflows: a balanced multiclass classification problem and a larger, mixed-feature regression problem. The Iris models tie at 0.933 accuracy on one 30-row test set. In the housing experiment, engineered-feature OLS has the lowest RMSE among six tested candidates, but its approximately $70,309 RMSE and capped area-level target make clear that the result is exploratory rather than a precise individual-home estimate.

## Internship Acknowledgement

These notebooks were completed as part of a Data Science Internship. The repository does not include an organization, mentor, or program name, so none is attributed here. I appreciate the opportunity to apply data analysis and machine learning methods to these two practical learning tasks.
