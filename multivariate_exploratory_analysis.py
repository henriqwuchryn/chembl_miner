import numpy as np
import pandas as pd
import seaborn as sns

from chembl_miner import data_preprocessing
import miscelanneous_methods as mm


sns.set_theme(style='ticks')
sns.set_theme(style='whitegrid', font='liberation serif')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import sys


try:
    filename = sys.argv[1]
except:
    print(
        '''\nyou must insert the dataset filename as an argument, like this:
    >python multivariate_exploratory_analysis.py FILENAME.csv''',
        )
    quit()

results_path = f'analysis/{filename[:-4]}'
datasets_path = 'datasets'

try:
    fingerprint_df = pd.read_csv(f'{datasets_path}/{filename}', index_col='index')
except:
    if not os.path.exists(f'{datasets_path}/{filename}'):
        print('File does not exist')
    else:
        print('Invalid file - cannot convert to dataframe')
    quit()

fingerprint_df = fingerprint_df.drop(
    fingerprint_df[fingerprint_df['bioactivity_class'] == 'intermediate'].index,
    )
features_df = fingerprint_df.drop(
    [
        'molecule_chembl_id',
        'neg_log_value',
        'bioactivity_class',
        ],
    axis=1,
    )

n_components = input('\nChoose how many principal components to compute\n')
n_components = mm.check_if_int(n_components, 10)
principal_components_colnames = []

for n in range(n_components):
    col_name = 'PC ' + str(n + 1)
    principal_components_colnames.append(col_name)

use_scaler = input('\nDo you want to use a scaler on the features data? 1 or 0\n')
use_scaler = mm.check_if_int(use_scaler)
if use_scaler == 1:
    features_df = data_preprocessing.scale_features(features_df, StandardScaler())
    results_path = f'{results_path}/scaled'
if not os.path.exists(results_path):
    os.makedirs(results_path)

pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(features_df)
pca_df = pd.DataFrame(
    data=principal_components,
    columns=principal_components_colnames,
    )
pca_df_classes = pd.concat(
    [
        pca_df,
        fingerprint_df['bioactivity_class'],
        ],
    axis=1,
    )
expl_var = pca.explained_variance_ratio_
print(
    '\nPrincipal components explained variance: ',
    expl_var,
    '\n',
    )
print(
    'Principal components total explained variance: ',
    expl_var.sum(),
    '\n',
    )

# scree_plot
plt.figure(figsize=(14, 10))
plt.plot(np.arange(1, len(expl_var) + 1), expl_var, marker='o')
plt.xlabel(
    'Principal Component',
    size=20,
    )
plt.ylabel(
    'Proportion of Variance Explained',
    size=20,
    )
plt.title(
    'Figure 1: Scree Plot for Proportion of Variance Explained',
    size=25,
    )
plt.grid(True)
plt.savefig(f'{results_path}/scree_plot-{n_components}_components.svg', bbox_inches="tight")
print(f'\nScree plot saved at {results_path}/scree_plot-{n_components}_components.svg')
plot_pcs = input(
    """\nAre you ok with the amount of principal components?
    Press 1 to plot every combination possible\n""",
    )
plot_pcs = mm.check_if_int(plot_pcs)

if plot_pcs == 1:

    def PcaGrapher(a: int, b: int, df: pd.DataFrame, expl_var: list) -> None:
        """plots two principal components from a given PCA dataframe containing a label column
        and saves them on a given path defined by global variable plots_path"""
        plt.figure(figsize=(8, 8))
        sns.jointplot(
            data=df, x=f"PC {str(a)}", y=f"PC {str(b)}",
            hue="bioactivity_class", hue_order=['active', 'inactive'],
            edgecolor='black', alpha=0.5, marginal_kws={'bw_adjust': 0.5},
            )
        expl_var_x = round(expl_var[a - 1] * 100, 2)
        expl_var_y = round(expl_var[b - 1] * 100, 2)
        plt.xlabel(f'PC {str(a)} (Explained variance: {str(expl_var_x)}%)', fontsize=16)
        plt.ylabel(f'PC {str(b)} (Explained variance: {str(expl_var_y)}%)', fontsize=16)
        plt.savefig(f'{results_path}/pca_{str(a)}_{str(b)}.svg', bbox_inches='tight')
        plt.close()


    print('\nPlotting every combination of principal components')
    for i in range(n_components):
        for j in range(n_components):
            if i != j:
                PcaGrapher(i + 1, j + 1, pca_df_classes, expl_var)
    print(f'\nPCA analysis are available at {results_path}')
