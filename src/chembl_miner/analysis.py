import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy.random import seed
from scipy.stats import mannwhitneyu
from sklearn.base import BaseEstimator
from statsmodels import api as sm
from rdkit import Chem
from rdkit.Chem import FilterCatalog
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity
from rdkit.ML.Cluster import Butina

from .datasets import TrainingData, PredictionData
from .utils import print_low, print_high


class DataExplorer:
    """
    A class for performing exploratory data analysis (EDA) on a TrainingData object.

    This class uses the training subset (x_train, y_train) for analysis to
    avoid leaking information from the test set.
    """


    def __init__(self, dataset: TrainingData):
        """
        Initializes the DataExplorer.

        Args:
            dataset (TrainingData): A TrainingData object containing the
                train/test splits.

        Example:
            ```python
            # Assuming 'my_dataset' is a populated TrainingData object
            explorer = DataExplorer(dataset=my_dataset)
            ```
        """
        self.general_data = dataset.subset_general_data(train_subset=True)
        self.target = dataset.y_train
        self.features = dataset.x_train


    # --- Univariate Plots ---

    def plot_target_distribution(self) -> plt.Figure:
        """
        Generates a histogram and KDE plot of the target variable.

        Displays the mean and median as vertical lines.

        Returns:
            plt.Figure: The Matplotlib Figure object for the plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(self.target, kde=True, ax=ax, bins=30)

        mean_val = self.target.mean()
        median_val = self.target.median()

        ax.axvline(mean_val, color='red', linestyle='--', lw=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='-', lw=2, label=f'Median: {median_val:.2f}')

        ax.set_title(f'Distribution of Target: {self.target.name}', fontsize=16)
        ax.set_xlabel(self.target.name, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        fig.tight_layout()
        return fig


    def plot_lipinski_descriptors(self) -> plt.Figure:
        """
        Generates box plots for key Lipinski descriptors.

        Plots 'MW', 'LogP', 'NumHDonors', and 'NumHAcceptors'.

        Returns:
            plt.Figure: The Matplotlib Figure object for the plot.
        """
        lipinski_cols = ['MW', 'LogP', 'NumHDonors', 'NumHAcceptors']
        if not all(col in self.general_data.columns for col in lipinski_cols):
            raise ValueError("Lipinski columns (MW, LogP, etc.) not found in the data.")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, col in enumerate(lipinski_cols):
            sns.boxplot(y=self.general_data[col], ax=axes[i])
            axes[i].set_title(f'Distribution of {col}', fontsize=14)
            axes[i].set_ylabel('Value', fontsize=12)

        fig.suptitle('Distribution of Lipinski Descriptors', fontsize=16, y=1.02)
        fig.tight_layout()
        return fig


    def plot_ro5_violations_vs_bioactivity(self) -> plt.Figure:
        """
        Generates a heatmap of Rule of 5 violations vs. bioactivity class.

        The heatmap shows the relative frequency of violation counts within
        each bioactivity class ('active', 'intermediate', 'inactive').

        Returns:
            plt.Figure: The Matplotlib Figure object for the plot.
        """
        required_cols = ['Ro5Violations', 'bioactivity_class']
        if not all(col in self.general_data.columns for col in required_cols):
            raise ValueError("Data must contain 'Ro5Violations' and 'bioactivity_class' columns.")

        # Create a cross-tabulation of the counts
        cross_counts = pd.crosstab(self.general_data['Ro5Violations'], self.general_data['bioactivity_class'])

        # Ensure a consistent column order
        cross_counts = cross_counts.reindex(columns=['active', 'intermediate', 'inactive'], fill_value=0)

        # Calculate the relative frequency for each column (bioactivity class)
        # This is a more efficient way to do the calculation from your original script
        relative_freq = cross_counts / cross_counts.sum(axis=0)

        # Generate the plot
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(data=relative_freq, annot=True, fmt=".2f", cmap='viridis', ax=ax)

        ax.set_title('Relative Frequency of Ro5 Violations by Bioactivity Class', fontsize=16)
        ax.set_xlabel('Bioactivity Class', fontsize=12)
        ax.set_ylabel('Number of Ro5 Violations', fontsize=12)
        fig.tight_layout()
        return fig


    def plot_lipinski_density(self, lipinski_descriptors: list[str] = None, group_by_bioactivity=True) -> plt.Figure:
        """
        Generates overlapping density plots for specified Lipinski descriptors.

        Args:
            lipinski_descriptors (list[str], optional): A list of descriptor
                columns to plot. Defaults to ['MW', 'LogP', 'NumHDonors',
                'NumHAcceptors'].
            group_by_bioactivity (bool, optional): If True, plots are colored
                by the 'bioactivity_class' column. Defaults to True.

        Returns:
            plt.Figure: The Matplotlib Figure object for the plot.

        Example:
            ```python
            explorer = DataExplorer(my_dataset)
            # Plot default descriptors grouped by bioactivity
            fig1 = explorer.plot_lipinski_density()
            
            # Plot only MW and LogP, not grouped
            fig2 = explorer.plot_lipinski_density(
                lipinski_descriptors=['MW', 'LogP'],
                group_by_bioactivity=False
            )
            ```
        """
        if lipinski_descriptors is None:
            lipinski_descriptors = ['MW', 'LogP', 'NumHDonors', 'NumHAcceptors']

        if not all(col in self.general_data.columns for col in lipinski_descriptors):
            raise ValueError(f"One or more specified descriptors not found in the data.")

        hue_column=None
        if group_by_bioactivity:
            if 'bioactivity_class' not in self.general_data.columns:
                raise ValueError(
                    "Argument 'group_by_bioactivity' is True, but the "
                    "'bioactivity_class' column was not found."
                    )
            else:
                hue_column = 'bioactivity_class'

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, desc in enumerate(lipinski_descriptors):
            sns.kdeplot(
                data=self.general_data,
                x=desc,
                ax=axes[i],
                hue=hue_column,
                fill=True,
                alpha=0.3,
                common_norm=False,
                hue_order=['active', 'intermediate', 'inactive'],
                palette={'active': 'forestgreen', 'intermediate': 'goldenrod', 'inactive': 'firebrick'},)
            axes[i].set_title(f'Density of {desc}', fontsize=14)
            axes[i].set_xlabel('Value', fontsize=12)
            axes[i].set_ylabel('Density', fontsize=12)

        fig.suptitle('Density of Lipinski Descriptors', fontsize=16, y=1.02)
        fig.tight_layout()
        return fig


    def plot_mw_vs_logp(self) -> plt.Figure:
        """
        Generates a scatter plot of Molecular Weight vs. LogP.

        Points are colored by bioactivity class, and Rule of 5 thresholds
        (MW=500, LogP=5) are drawn as dashed lines.

        Returns:
            plt.Figure: The Matplotlib Figure object for the plot.
        """
        required_cols = ['MW', 'LogP', 'bioactivity_class']
        if not all(col in self.general_data.columns for col in required_cols):
            raise ValueError("Data must contain 'MW', 'LogP', and 'bioactivity_class' columns.")

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(
            data=self.general_data,
            x='MW',
            y='LogP',
            hue='bioactivity_class',
            ax=ax,
            alpha=0.7,
            s=50,
            hue_order=['active', 'intermediate', 'inactive'],
            palette={'active': 'forestgreen', 'intermediate': 'goldenrod', 'inactive': 'firebrick'},
            )

        # Add Rule of 5 threshold lines
        ax.axvline(x=500, color='black', linestyle='--', lw=2, label='MW = 500')
        ax.axhline(y=5, color='black', linestyle=':', lw=2, label='LogP = 5')

        ax.set_title('Molecular Weight vs. LogP', fontsize=16)
        ax.set_xlabel('Molecular Weight (MW)', fontsize=12)
        ax.set_ylabel('LogP', fontsize=12)
        ax.legend(title='Bioactivity Class')
        fig.tight_layout()
        return fig


    # --- Multivariate Plots ---

    def plot_target_vs_feature(self, feature_name: str) -> plt.Figure:
        """
        Generates a plot of the target variable against a single feature.

        - A scatter plot is used for numeric features (with > 15 unique values).
        - A box plot is used for categorical features.

        Args:
            feature_name (str): The name of the feature column to plot.

        Returns:
            plt.Figure: The Matplotlib Figure object for the plot.

        Example:
            ```python
            explorer = DataExplorer(my_dataset)
            
            # Plot against a categorical feature (e.g., a fingerprint bit)
            fig1 = explorer.plot_target_vs_feature(feature_name='PubchemFP10')
            
            # Plot against a numeric feature
            fig2 = explorer.plot_target_vs_feature(feature_name='Ro5Violations')
            ```
        """
        if feature_name in self.features.columns:
            feature = self.features[feature_name]
        elif feature_name in self.general_data.columns:
            feature = self.general_data[feature_name]
        else:
            raise ValueError(f"Feature '{feature_name}' not found in the data.")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Check if feature is numeric or categorical to decide the plot type
        if pd.api.types.is_numeric_dtype(feature) and feature.nunique() > 15:
            sns.scatterplot(x=feature, y=self.target, ax=ax, alpha=0.5)
            ax.set_title(f'{self.target.name} vs. {feature_name} (Scatter Plot)', fontsize=16)
        else:
            sns.boxplot(x=feature, y=self.target, ax=ax)
            ax.set_title(f'{self.target.name} vs. {feature_name} (Box Plot)', fontsize=16)
            ax.tick_params(axis='x', rotation=45)

        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel(self.target.name, fontsize=12)
        fig.tight_layout()
        return fig


    def plot_correlation_heatmap(self, top_n: int = 15) -> plt.Figure:
        """
        Generates a heatmap of the features most correlated with the target.

        This plot shows the correlation matrix for the top N features
        (by absolute correlation) and the target variable.

        Args:
            top_n (int, optional): The number of top features to include
                in the heatmap. Defaults to 15.

        Returns:
            plt.Figure: The Matplotlib Figure object for the plot.

        Example:
            ```python
            explorer = DataExplorer(my_dataset)
            fig = explorer.plot_correlation_heatmap(top_n=20)
            fig.savefig("correlation_heatmap.png")
            ```
        """
        numeric_features = self.features.select_dtypes(include=np.number)
        df_corr = pd.concat([numeric_features, self.target], axis=1).corr()

        # Get the top N features most correlated with the target
        top_corr_cols = df_corr[self.target.name].abs().sort_values(ascending=False).head(top_n + 1).index
        top_corr_matrix = df_corr.loc[top_corr_cols, top_corr_cols]

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(top_corr_matrix, dtype=bool))  # Mask for upper triangle
        sns.heatmap(top_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', mask=mask, ax=ax)

        ax.set_title(f'Top {top_n} Features Correlated with {self.target.name}', fontsize=16)
        fig.tight_layout()
        return fig


class ModelAnalyzer:
    """
    A class for diagnosing and analyzing a fitted machine learning model.

    Generates common diagnostic plots for regression models, such as
    residuals vs. fitted, Q-Q plots, and actual vs. predicted.
    """


    def __init__(
        self,
        fit_model: BaseEstimator,
        dataset: TrainingData,
        algorithm_name: str = None,
        train_subset=True,
        ):
        """
        Initializes the ModelAnalyzer.

        Args:
            fit_model (BaseEstimator): A fitted scikit-learn compatible model.
            dataset (TrainingData): The dataset used for training/testing.
            algorithm_name (str, optional): A name for the algorithm. If None,
                the model's class name is used. Defaults to None.
            train_subset (bool, optional): If True, analysis is performed on
                the training set. If False, it's performed on the test set.
                Defaults to True.
        
        Example:
            ```python
            # Assuming 'my_model' is a fitted model (e.g., from ml.fit_model)
            # and 'my_dataset' is a TrainingData object.
            
            # Analyze performance on the test set
            analyzer_test = ModelAnalyzer(
                fit_model=my_model,
                dataset=my_dataset,
                train_subset=False 
            )
            
            # Analyze performance on the training set
            analyzer_train = ModelAnalyzer(
                fit_model=my_model,
                dataset=my_dataset,
                train_subset=True
            )
            ```
        """
        self.model = fit_model
        self.dataset = dataset
        self.algorithm_name = fit_model.__class__.__name__ if algorithm_name is None else algorithm_name

        # Pre-calculate common values to use in all plots
        if train_subset:
            features = self.dataset.x_train
            self.y_true = self.dataset.y_train
            self.y_pred = self.model.predict(features)

        else:
            features = self.dataset.x_test
            self.y_true = self.dataset.y_test
            self.y_pred = self.model.predict(features)
        self.residuals = self.y_true - self.y_pred

        # Pre-calculate studentized residuals for advanced plots
        try:
            ols_model = sm.OLS(self.y_true, sm.add_constant(features)).fit()
            influence = ols_model.get_influence()
            self.residuals_std = influence.resid_studentized_internal
        except Exception:
            self.residuals_std = (self.residuals - self.residuals.mean()) / self.residuals.std()


    def plot_residuals_vs_fitted(self) -> plt.Figure:
        """
        Generates a Residuals vs. Fitted plot.

        Used to check for non-linear patterns and heteroscedasticity.

        Returns:
            plt.Figure: The Matplotlib Figure object for the plot.
        """
        fig, ax = plt.subplots(figsize=(8, 7))
        # ... (plotting code from the previous _plot helper) ...
        sns.residplot(
            x=self.y_pred, y=self.residuals, lowess=True, ax=ax,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red', 'lw': 2, 'label': 'Trend'},
            )
        ax.set_title('Residuals vs. Fitted Values', fontsize=14)
        ax.set_xlabel('Fitted Values (y_pred)', fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.legend()
        fig.tight_layout()
        return fig


    def plot_qq(self) -> plt.Figure:
        """
        Generates a Normal Q-Q plot.

        Used to check if the residuals are normally distributed.

        Returns:
            plt.Figure: The Matplotlib Figure object for the plot.
        """
        fig, ax = plt.subplots(figsize=(8, 7))
        # ... (plotting code from the previous _plot helper) ...
        sm.qqplot(self.residuals, line='s', ax=ax, alpha=0.5)
        ax.lines[1].set_color('red')
        ax.lines[1].set_linewidth(2)
        ax.set_title('Normal Q-Q Plot', fontsize=14)
        fig.tight_layout()
        return fig


    def plot_scale_location(self) -> plt.Figure:
        """
        Generates a Scale-Location plot.

        Used to check for homoscedasticity (if the variance of residuals
        is constant across the range of fitted values).

        Returns:
            plt.Figure: The Matplotlib Figure object for the plot.
        """
        fig, ax = plt.subplots(figsize=(8, 7))
        sqrt_std_resid = np.sqrt(np.abs(self.residuals_std))
        sns.scatterplot(x=self.y_pred, y=sqrt_std_resid, ax=ax, alpha=0.5)
        sns.regplot(
            x=self.y_pred, y=sqrt_std_resid, scatter=False, lowess=True, ax=ax,
            line_kws={'color': 'red', 'lw': 2, 'label': 'Trend'},
            )
        ax.set_title('Scale-Location Plot', fontsize=14)
        ax.set_xlabel('Fitted Values', fontsize=12)
        ax.set_ylabel('$\\sqrt{|Standardized \, Residuals|}$', fontsize=12)
        ax.legend()
        fig.tight_layout()
        return fig


    def plot_actual_vs_predicted(self) -> plt.Figure:
        """
        Generates an Actual vs. Predicted plot.

        Used to assess the overall accuracy of the model. A perfect model
        would have all points on the 'Perfect Fit' line.

        Returns:
            plt.Figure: The Matplotlib Figure object for the plot.
        """
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.scatterplot(x=self.y_true, y=self.y_pred, ax=ax, alpha=0.6)
        sns.regplot(
            x=self.y_true, y=self.y_pred, scatter=False, lowess=True, ax=ax,
            line_kws={'color': 'red', 'lw': 2, 'label': 'Trend'},
            )
        # Expected line (perfect prediction)
        limits = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
            ]
        ax.plot(limits, limits, color='black', linestyle='--', lw=2, label='Perfect Fit (y_true=y_pred)')
        ax.set_title('Actual vs. Predicted Values', fontsize=14)
        ax.set_xlabel('Actual Values (y_true)', fontsize=12)
        ax.set_ylabel('Predicted Values (y_pred)', fontsize=12)
        ax.legend()
        fig.tight_layout()
        return fig


    def plot_residuals_by_id(self, top_n: int = 30) -> plt.Figure:
        """
        Generates a bar plot of the top N largest absolute residuals.

        Used to identify specific samples (by their index/ID) that the
        model struggles to predict accurately.

        Args:
            top_n (int, optional): The number of top residuals to display.
                Defaults to 30.

        Returns:
            plt.Figure: The Matplotlib Figure object for the plot.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        abs_residuals = np.abs(self.residuals).sort_values(ascending=False).head(top_n)
        ids = [str(i) for i in abs_residuals.index]  # Ensure IDs are strings
        sns.barplot(x=ids, y=abs_residuals.values, ax=ax, color='lightcoral')
        ax.axhline(0, color='black', lw=1)
        ax.set_title(f'Top {top_n} Absolute Residuals by ID', fontsize=14)
        ax.set_xlabel('Sample ID / Index', fontsize=12)
        ax.set_ylabel('Absolute Residual', fontsize=12)
        ax.tick_params(axis='x', rotation=90)
        fig.tight_layout()
        return fig


    # TODO: add model explainers such as shap
    def plot_shap_summary(self):
        """
        (Future) Generates a SHAP summary plot.
        """
        print("SHAP functionality not yet implemented.")
        # import shap
        # explainer = shap.TreeExplainer(self.model)
        # shap_values = explainer.shap_values(self.dataset.x_test)
        # shap.summary_plot(shap_values, self.dataset.x_test)
        pass


def mannwhitney_test(col_name: str, molecules_df1, molecules_df2, alpha: float = 0.05):
    """
    Performs a Mann-Whitney U test on a specific column from two DataFrames.

    Args:
        col_name (str): The name of the column to test.
        molecules_df1 (pd.DataFrame): The first group's data.
        molecules_df2 (pd.DataFrame): The second group's data.
        alpha (float, optional): The significance level. Defaults to 0.05.

    Returns:
        pd.DataFrame: A DataFrame containing the test statistics, p-value,
            and interpretation.
    
    Example:
        ```python
        # Create two mock dataframes
        group_a = pd.DataFrame({'MW': [300, 310, 305, 315]})
        group_b = pd.DataFrame({'MW': [450, 455, 460, 458]})
        
        results = mannwhitney_test('MW', group_a, group_b)
        print(results)
        ```
    """
    # Inspirado em: https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
    seed(1)
    col1 = molecules_df1[col_name]
    col2 = molecules_df2[col_name]
    stat, p = mannwhitneyu(col1, col2)

    if p > alpha:
        interpretation = 'Same distribution (fail to reject H0)'
    else:
        interpretation = 'Different distributions (reject H0)'

    results = pd.DataFrame(
        {
            'Descriptor'    : col_name,
            'Statistics'    : stat,
            'p'             : p,
            'alpha'         : alpha,
            'Interpretation': interpretation,
            }, index=[0],
        )
    # filename = 'mannwhitneyu_' + descriptor + '.csv'
    # results.to_csv(filename,index=False)
    return results


def filter_by_pains(
    prediction_data: PredictionData,
    smiles_col: str = 'canonical_smiles',
    ) -> pd.Series:
    """
    Screens molecules for PAINS (Pan-Assay Interference Compounds).

    This function modifies `prediction_data.deploy_data` in-place by
    adding (or overwriting) an 'is_pains' column.

    Args:
        prediction_data (PredictionData): The dataset object containing
            molecules to screen.
        smiles_col (str, optional): The name of the column in
            `prediction_data.deploy_data` that contains the SMILES strings.
            Defaults to 'canonical_smiles'.

    Returns:
        pd.Series: A boolean Series where True indicates a potential
            PAINS compound.
    
    Example:
        ```python
        # Assuming 'deploy_data' is a PredictionData object
        # with a 'canonical_smiles' column
        
        # This modifies deploy_data.deploy_data in-place
        pains_series = filter_by_pains(deploy_data)
        
        print(deploy_data.deploy_data['is_pains'].value_counts())
        ```
    """
    if prediction_data.deploy_data.empty:
        print("PredictionData.deploy_data is empty. Returning an empty Series.")
        return pd.Series(dtype=bool)

    if smiles_col not in prediction_data.deploy_data.columns:
        raise ValueError(
            f"SMILES column '{smiles_col}' not found in prediction_data.deploy_data."
            )

    print_low(f"Screening {len(prediction_data.deploy_data)} molecules for PAINS.")

    # Initialize the PAINS filter catalog from RDKit
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog.FilterCatalog(params)

    pains_results = []
    for smiles in prediction_data.deploy_data[smiles_col]:
        if not smiles or not isinstance(smiles, str):
            pains_results.append(False)  # Treat invalid/missing SMILES as non-PAINS
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            pains_results.append(False)  # Treat invalid SMILES as non-PAINS
            continue

        # Check for matches
        if catalog.HasMatch(mol):
            pains_results.append(True)
        else:
            pains_results.append(False)

    num_pains = sum(pains_results)
    print_high(f"Found {num_pains} potential PAINS compounds.")
    pains_results_series = pd.Series(
        pains_results,
        index=prediction_data.deploy_data.index,
        name="is_pains",
        )
    if "is_pains" in prediction_data.deploy_data.columns:
        print_high(f"Warning: Column 'is_pains' already exists. It will be overwritten.")
        # Drop old column to prevent join issues
        prediction_data.deploy_data = prediction_data.deploy_data.drop(
            columns=["is_pains"]
        )
    prediction_data.deploy_data['is_pains'] = pains_results_series
    return pains_results_series



def assign_molecule_clusters(
        prediction_data: PredictionData,
        smiles_col: str = 'canonical_smiles',
        cluster_col_name: str = 'cluster_id',
        similarity_cutoff: float = 0.7,
        radius: int = 2,
        fingerprint_n_bits: int = 1024,
) -> None:
    """
    Clusters molecules using Butina clustering and assigns a cluster ID.

    This function modifies the `prediction_data.deploy_data` DataFrame in-place
    by adding a new column with the cluster ID.

    Args:
        prediction_data (PredictionData): The dataset object to modify.
        smiles_col (str, optional): The column in `deploy_data` with SMILES
            strings. Defaults to 'canonical_smiles'.
        cluster_col_name (str, optional): The name for the new cluster ID
            column. Defaults to 'cluster_id'.
        similarity_cutoff (float, optional): The Tanimoto similarity threshold
            for clustering (1.0 - distance_cutoff). Defaults to 0.7.
        radius (int, optional): The radius for the Morgan fingerprint.
            Defaults to 2.
        fingerprint_n_bits (int, optional): The number of bits for the
            Morgan fingerprint. Defaults to 1024.
    
    Example:
        ```python
        # Assuming 'deploy_data' is a PredictionData object
        # with a 'canonical_smiles' column
        
        assign_molecule_clusters(
            deploy_data,
            cluster_col_name='scaffold_cluster',
            similarity_cutoff=0.6
        )
        
        print(deploy_data.deploy_data['scaffold_cluster'].value_counts())
        ```
    """
    print_low(f"Assigning molecule clusters to '{cluster_col_name}' column.")

    if smiles_col not in prediction_data.deploy_data.columns:
        raise ValueError(
            f"SMILES column '{smiles_col}' not found in prediction_data.deploy_data."
        )

    data_to_cluster = prediction_data.deploy_data
    print_high(f"Clustering {len(data_to_cluster)} molecules.")

    # --- 1. Generate Fingerprints ---
    mols = []
    valid_indices = []  # To map list index (0,1,2..) back to DataFrame index
    for idx, smiles in data_to_cluster[smiles_col].items():
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mols.append(mol)
            valid_indices.append(idx)
        else:
            print_high(f"Warning: Could not parse SMILES at index {idx}. Skipping.")

    if not mols:
        print_low("No valid molecules found to cluster.")
        return

    fingerprint_generator = GetMorganGenerator(radius=radius, fpSize=fingerprint_n_bits)
    fingerprints = fingerprint_generator.GetFingerprints(mols)

    # --- 2. Run Butina Clustering ---
    distances = []
    n_mols = len(fingerprints)
    for i in range(n_mols):
        similarity_values = BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        distances.extend([1 - value for value in similarity_values])

    clusters = Butina.ClusterData(
        distances,
        n_mols,
        1.0 - similarity_cutoff,  # Butina uses distance cutoff
        isDistData=True,
    )
    clusters = sorted(clusters, key=len, reverse=True)
    print_high(f"Clustered {n_mols} valid molecules into {len(clusters)} clusters.")

    # --- 3. Map clusters back to data ---
    cluster_id_map = {}  # Dict to map {df_index: cluster_id}
    for cluster_id, mol_indices in enumerate(clusters):
        original_indices = [valid_indices[i] for i in mol_indices]
        for original_idx in original_indices:
            cluster_id_map[original_idx] = cluster_id

    cluster_map_series = pd.Series(cluster_id_map, name=cluster_col_name, dtype='Int64')

    # --- 4. Assign new column in-place ---
    if cluster_col_name in prediction_data.deploy_data.columns:
        print_high(f"Warning: Column '{cluster_col_name}' already exists. It will be overwritten.")
        # Drop old column to prevent join issues
        prediction_data.deploy_data = prediction_data.deploy_data.drop(
            columns=[cluster_col_name]
        )

    # Join the series back to the original dataframe
    prediction_data.deploy_data[cluster_col_name] = cluster_map_series

    # Note: Molecules with invalid SMILES will have NaN in this new column.

    print_low(f"Cluster assignments complete. Column '{cluster_col_name}' added to deploy_data.")
    return