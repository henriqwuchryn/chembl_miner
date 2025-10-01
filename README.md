# ChEMBL Miner

`chembl_miner` is a Python package designed to streamline the process of Quantitative Structure-Activity Relationship (QSAR)
analysis using data from the [ChEMBL database](https://www.ebi.ac.uk/chembl/). It provides a comprehensive suite of tools for data retrieval, preprocessing,
feature engineering, machine learning, and model interpretation, enabling a complete and automated workflow.

## Key Features & Utilities

* **Data Retrieval**: Fetch bioactivity data from the ChEMBL database using target ChEMBL IDs and specific activity types.
* **Data Preprocessing**: A robust and customizable preprocessing pipeline to clean and prepare your data for modeling. This includes handling missing values, converting units, and treating duplicate entries.
* **Feature Engineering**: Calculate various molecular fingerprints from [PaDEL descriptors](http://yapcwsoft.com/dd/padeldescriptor/) (e.g., PubChem, MACCS, EState) to be used as features in your machine learning models.
* **Machine Learning**: A flexible wrapper for scikit-learn compatible regression models. It handles hyperparameter optimization using a genetic algorithm, model evaluation with cross-validation, and fitting of the final model.
* **Dataset Management**: Convenient wrappers to handle and split your datasets for both training and deployment. It supports structural (or scaffold) splitting to avoid data leakage from the test subset into the training subset.
* **Model Explanation and Analysis**: Tools for both exploratory data analysis and model interpretation, allowing you to understand your data and the predictions of your trained models.

---

## Typical Workflow

The package is designed to be used in a sequential workflow:

1.  **Data Retrieval**: Fetch activity data from ChEMBL using `get_activity_data`.
2.  **Assay Review (Optional)**: If needed, review and filter specific assays with `review_assays`.
3.  **Preprocessing**: Clean the retrieved data with the `preprocess_data` function.
4.  **Feature Calculation**: Generate molecular fingerprints using `calculate_fingerprint`.
5.  **Dataset Creation**: Create a `TrainingData` object from your preprocessed data and calculated features. This will handle the splitting of data into training and test sets.
6.  **Model Training and Evaluation**:
    * Set up a machine learning model using the `ModelPipeline`.
    * Optimize the model's hyperparameters.
    * Evaluate the model's performance using cross-validation.
    * Fit the final model on the entire training set.
7.  **Analysis and Interpretation**:
    * Predict using a fit model on a `PredictionData` object built from deployment data.
    * Explore your dataset with `DataExplorer`.
    * Interpret your fitted model's behavior with `ModelAnalyzer`.

---

## Credits

This package's functionality was developed using several open source packages:
* pandas
* numpy
* scipy
* scikit-learn
* sklearn-genetic-opt
* matplotlib
* seaborn
* chembl_webresource_client
* padelpy
* rdkit
* xgboost
* statsmodels

---

## Future Implementations Roadmap


* **Expanded Descriptor Support**: Generalize the feature engineering module to support a wider range of molecular descriptors beyond fingerprints.
* **Advanced Hyperparameter Search**: Implement additional hyperparameter optimization methods, such as Grid Search and Random Search.
* **Enhanced CV Visualization**: Add functionality to better visualize cross-validation results to assess model generalization.
* **Classification Models**: Incorporate support for classification algorithms in addition to the current regression models.
* **Model Explainability**: Integrate SHAP (SHapley Additive exPlanations) for more in-depth model interpretation.
* **Similarity Filtering**: Add options for similarity-based filtering during the initial data retrieval step.
* **R Implementation**: Explore the possibility of creating an R version of the package.