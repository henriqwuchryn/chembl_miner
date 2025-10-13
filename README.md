# `chembl_miner`: A Python Package for QSAR Analysis

`chembl_miner` is a Python package that streamlines the process of Quantitative Structure-Activity Relationship (QSAR) analysis using data from the [ChEMBL database](https://www.ebi.ac.uk/chembl/). It provides a complete and automated workflow with a suite of tools for data retrieval, preprocessing, feature engineering, machine learning, and model interpretation.

## Motivation 🤔

This package was developed to organize and automate the QSAR analysis workflow commonly used in pharmaceutical sciences for drug discovery and repurposing. The goal is to provide a clean, reusable, and shareable tool for researchers in the field.

## Target Audience 🎯

The primary audience for this package is drug discovery and repurposing researchers, including medicinal chemists, bioinformaticians, and data scientists.

## Key Features 🔬

  * **Data Retrieval**: Fetch bioactivity data from the ChEMBL database using target ChEMBL IDs and specific activity types.
  * **Data Preprocessing**: A robust and customizable preprocessing pipeline to clean and prepare your data for modeling, which includes handling missing values, converting units, and treating duplicate entries.
  * **Feature Engineering**: Calculate various molecular fingerprints from [PaDEL descriptors](http://yapcwsoft.com/dd/padeldescriptor/) (e.g., PubChem, MACCS, EState).
  * **Machine Learning**: A flexible wrapper for scikit-learn compatible regression models that handles hyperparameter optimization using a genetic algorithm, model evaluation with cross-validation, and fitting of the final model.
  * **Dataset Management**: Convenient wrappers to handle and split datasets for both training and deployment, with support for structural (or scaffold) splitting.
  * **Model Explanation and Analysis**: Tools for both exploratory data analysis and model interpretation.

## Installation 💻

You can install `chembl_miner` using pip. There are no other dependencies required.

```bash
pip install chembl_miner
```

## Typical Workflow 🧪

The following is a general workflow for using `chembl_miner`.

### 1\. Data Retrieval and Preprocessing

First, fetch the activity data from ChEMBL for a specific target and preprocess it.

```python
from chembl_miner import *
import pandas as pd

# Set verbosity level (0, 1, or 2)
set_verbosity(1)

# Fetch activity data from ChEMBL
target_chembl_id = "CHEMBL203"  # Example: Cyclooxygenase-1
activity_df = get_activity_data(target_chembl_id, activity_type="IC50")

# Optionally, review and filter assays. For example, to exclude assays with certain keywords:
id_list = review_assays(activity_df, max_entries=5,
                        assay_keywords=['mutant', 'mutated'], exclude_keywords=True)

# Preprocess the data
activity_df = preprocess_data(activity_df, convert_units=True, assay_ids=id_list)

# Calculate molecular fingerprints
descriptors_df = calculate_fingerprint(activity_df, fingerprint="pubchem")

# Create a TrainingData object
# This will split the data into training and testing sets.
# You can use any TrainingData object for the subsequent steps.
dataset = TrainingData.from_dataframe(activity_df, descriptors_df, use_structural_split=True)

# Saving the dataset is optional but recommended for reproducibility
# and to avoid re-running the preprocessing steps.
dataset.to_path("my_qsar_dataset")
```

#### Example Output of `review_assays`

```
Displaying 5 of 1894 total unique assays.
To see more, adjust the 'max_entries' parameter.

assay_chembl_id  assay_description
CHEMBL648388     In vitro antifungal activity against Aspergillus fumigatus                189
CHEMBL647385     Minimum inhibitory concentration (MIC) against Aspergillus fumigatus          116
CHEMBL3266263    Antimicrobial activity against Aspergillus fumigatus 7544 after 48 hrs...   111
CHEMBL649169     Antifungal activity against Aspergillus fumigatus Saito strain                103
CHEMBL899327     Antifungal activity against Aspergillus fumigatus                              98
Name: count, dtype: int64
```

### 2\. Hyperparameter Optimization

Next, optimize the hyperparameters of your machine learning model. This example uses XGBoost regression and optimizes for Mean Absolute Error (MAE).

```python
# Load the dataset (if you saved it previously)
# dataset = TrainingData.from_path('my_qsar_dataset')

# Set up the model pipeline
ml = ModelPipeline.setup('xgboost_reg', scoring=['mae'])

# Optimize hyperparameters using a genetic algorithm
param_search = ml.optimize_hyperparameters(dataset=dataset, cv=5, refit='mae', population_size=40)

# Evaluate the model with 10-fold cross-validation
cv_results = ml.evaluate_model(dataset=dataset, cv=10)
print(ml.unpack_cv_results(cv_results))

# The best parameters are stored in ml.params
print('Best parameters found:', ml.params)
```

#### Example Output of `unpack_cv_results`

```
  scorer dataset_type      mean        sd
0    mae         test -0.486349  0.063903
1    mae        train -0.103694  0.003171
```

### 3\. Model Training and Analysis

With the best hyperparameters, train the final model and analyze its performance. For a detailed guide on interpreting the diagnostic plots, please refer to [this resource](https://ema.drwhy.ai/residualDiagnostic.html) by Przemyslaw Biecek and Tomasz Burzykowski (2020).

```python
from chembl_miner import ModelAnalyzer
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# If `refit` was used in `optimize_hyperparameters`, the model is already fitted.
# Otherwise, you can fit it with the best parameters:
# ml.fit(dataset, params=ml.params)

# Align test set features with the model's features
dataset.x_test = dataset.x_test[ml.fit_model.feature_names_in_]

# Make predictions on the test set
y_pred = ml.fit_model.predict(dataset.x_test)

# Calculate performance metrics
r2 = r2_score(y_true=dataset.y_test, y_pred=y_pred)
mae = mean_absolute_error(y_true=dataset.y_test, y_pred=y_pred)
rmse = root_mean_squared_error(y_true=dataset.y_test, y_pred=y_pred)

print(f"R^2 Score: {r2:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Analyze the model's performance on the test set
explainer = ModelAnalyzer(dataset=dataset, fit_model=ml.fit_model, train_subset=False)
explainer.plot_actual_vs_predicted()
explainer.plot_residuals_vs_fitted()
```

## Future Implementations 🚀

  - **Expanded Descriptor Support**: Addition of more molecular descriptors (e.g.: RDKit fingerprinters).
  - **Advanced Hyperparameter Search**: Implementation of additional methods like Grid Search and Random Search.
  - **Enhanced CV Visualization**: Improved visualization of cross-validation results.
  - **Classification Models**: Support for classification algorithms.
  - **Model Explainability**: Integration of SHAP for deeper model interpretation.
  - **Similarity Filtering**: Options for similarity-based filtering during data retrieval.
  - **R Implementation**: Potential for an R version of the package.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## Issues and Contact 📬

Please report any issues or questions on the [GitHub issues page](https://github.com/henriqwuchryn/chembl_miner/issues) or through e-mail: henrique.wuchryn@ufpr.br.

```
```
