import molecules_manipulation_methods as mmm
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme(style='ticks')
sns.set_theme(style='whitegrid',font='times new roman')
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import sys

try:
    filename = sys.argv[1]
except:
    print(
        '''you must insert the dataset filename as an argument, like this:
    >python multivariate_exploratory_analysis.py FILENAME.csv'''
    )
    quit()

results_path = f'analysis/{filename}'
datasets_path = 'datasets'

try:
    fingerprint_df = pd.read_csv(f'{datasets_path}/{filename}')
except:
    if not os.path.exists(f'{datasets_path}/{filename}'):
        print('File does not exist')
    else:
        print('Invalid file - cannot convert to dataframe')
    quit()
if not os.path.exists(results_path):
    os.makedirs(results_path)

features_df = fingerprint_df.drop(['molecule_chembl_id',
                                    'neg_log_value',
                                    'bioactivity_class'],
                                    axis=1)

n_components = input('Choose how many principal components to compute')
n_components = mmm.check_if_int(n_components,10)
principal_components_colnames = []

for n in range(n_components):
    col_name = 'PC '+str(n)
    principal_components_colnames.append(col_name)

pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(features_df)
pca_df = pd.DataFrame(data=principal_components,
                    columns= principal_components_colnames)
pca_df_classes = pd.concat([pca_df,
                        fingerprint_df['bioactivity_class']],
                        axis=1)
print('Principal components explained variance: ',
    pca.explained_variance_ratio_,
    '\n')
print('Principal components total explained variance: ',
    pca.explained_variance_ratio_.sum(),
    '\n')

#scree_plot
plt.figure(figsize=(14,10))
plt.plot(np.arange(1, len(pca.explained_variance_ratio_)+1), 
                   pca.explained_variance_ratio_, marker='o')
plt.xlabel('Principal Component',
           size = 20)
plt.ylabel('Proportion of Variance Explained',
           size = 20)
plt.title('Figure 1: Scree Plot for Proportion of Variance Explained',
          size = 25)
plt.grid(True)
plt.savefig(f'{results_path}/scree_plot-{n_components}_components.svg', bbox_inches="tight")
plot_pcs = input("""Are you ok with the amount of principal components?
Press 1 to plot every combination possible""")
plot_pcs = mmm.check_if_int(plot_pcs)

if plot_pcs == 1:

    def PcaGrapher(a:int,b:int,df:pd.DataFrame) -> None:
        """plots two principal components from a given PCA dataframe containing a label column
        and saves them on a given path defined by global variable plots_path"""
        plt.figure(figsize=(9,9))
        sns.scatterplot(data=df, 
                    x=f"PC {str(a)}", 
                    y=f"PC {str(b)}",
                    hue="bioactivity_class",
                    style="bioactivity_class",
                    style_order=["active","inactive","intermediate"],            
                    s=40)    
        plt.xlabel("PC "+str(a)+" ("+str(round(pca.explained_variance_ratio_[a-1]*100, 2))+"%)",
                    fontsize=16)
        plt.ylabel("PC "+str(b)+" ("+str(round(pca.explained_variance_ratio_[b-1]*100, 2))+"%)",
                    fontsize=16)
        plt.savefig(f'{results_path}/pca_{str(a)}_{str(b)}.svg',bbox_inches='tight')
        plt.close()

    for i in range (n_components):
        for j in range (n_components):
            if i != j:
                PcaGrapher(i+1,j+1,cryptococcus_pca)
