import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy.random import seed
from scipy.stats import mannwhitneyu
from sklearn.base import BaseEstimator
from statsmodels import api as sm

from .datasets import TrainingData


class DataExplorer:
    """
    A class for performing exploratory data analysis on a dataset.

    It can be initialized from a simple DataFrame or a DatasetWrapper.
    """


    def __init__(self, dataset: TrainingData):
        """
        Constructor method to create an explainer from a DatasetWrapper object.

        Note: This will use the training data (x_train, y_train) for analysis, to avoid leaking data from test subset.
        """
        self.general_data = dataset.subset_general_data(train_subset=True)
        self.target = dataset.y_train
        self.features = dataset.x_train


    # --- Univariate Plots ---

    def plot_target_distribution(self) -> plt.Figure:
        """
        Generates a histogram and KDE plot of the target variable.
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
        Generates a heatmap of the relative frequency of Rule of 5 violations
        within each bioactivity class.
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


    def plot_lipinski_density(self, lipinski_descriptors: list[str] = None) -> plt.Figure:
        """
        Generates overlapping density plots for specified Lipinski descriptors.

        Args:
            lipinski_descriptors (list[str]): A list of descriptor columns to plot.
                                               Defaults to ['MW', 'LogP', 'NumHDonors', 'NumHAcceptors'].
        """
        if lipinski_descriptors is None:
            lipinski_descriptors = ['MW', 'LogP', 'NumHDonors', 'NumHAcceptors']

        if not all(col in self.general_data.columns for col in lipinski_descriptors):
            raise ValueError(f"One or more specified descriptors not found in the data.")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, desc in enumerate(lipinski_descriptors):
            sns.kdeplot(self.general_data[desc], ax=axes[i], fill=True, alpha=0.5)
            axes[i].set_title(f'Density of {desc}', fontsize=14)
            axes[i].set_xlabel('Value', fontsize=12)
            axes[i].set_ylabel('Density', fontsize=12)

        fig.suptitle('Density of Lipinski Descriptors', fontsize=16, y=1.02)
        fig.tight_layout()
        return fig


    def plot_mw_vs_logp(self) -> plt.Figure:
        """
        Generates a scatter plot of Molecular Weight vs. LogP.

        Points are colored by bioactivity class, with lines indicating Ro5 thresholds.
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

        - Scatter plot for numeric features.
        - Box plot for categorical features.
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
    A class for explaining and diagnosing a fitted machine learning model.
    """


    def __init__(
        self,
        fit_model: BaseEstimator,
        dataset: TrainingData,
        algorithm_name: str = None,
        train_subset=True,
        ):
        """
        Initializes the explainer with a fitted model, DatasetWrapper object and optionally, algorithm_name.
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
        """Generates a Residuals vs. Fitted plot.

        Returns:
            plt.Figure: The Matplotlib Figure object for the plot."""
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
        """Generates a Normal Q-Q plot.

        Returns:
            plt.Figure: The Matplotlib Figure object for the plot."""
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
        Generates a Scale-Location plot to check for homoscedasticity.

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
        Generates an Actual vs. Predicted plot to assess model accuracy.

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

        Args:
            top_n (int): The number of top residuals to display.

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
        """(Future) Generates a SHAP summary plot."""
        print("SHAP functionality not yet implemented.")
        # import shap
        # explainer = shap.TreeExplainer(self.model)
        # shap_values = explainer.shap_values(self.dataset.x_test)
        # shap.summary_plot(shap_values, self.dataset.x_test)
        pass


def mannwhitney_test(col_name: str, molecules_df1, molecules_df2, alpha: float = 0.05):
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
