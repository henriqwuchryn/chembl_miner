import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme(style='ticks')
sns.set_theme(style='whitegrid',font='liberation serif')
import matplotlib.pyplot as plt
import molecules_manipulation_methods as mmm
import os
import sys

try:
    filename = sys.argv[1]
except:
    print(
        '''\nyou must insert the dataset filename as an argument, like this:
    >python univariate_exploratory_analysis.py FILENAME.csv'''
    )
    quit()

results_path = f'analysis/{filename[:-4]}'
datasets_path = 'datasets'

try:
    activity_df = pd.read_csv(f'{datasets_path}/{filename}')
except:
    if not os.path.exists(f'{datasets_path}/{filename}'):
        print('\nFile does not exist')
    else:
        print('\nInvalid file - cannot convert to dataframe')
    quit()
if not os.path.exists(results_path):
    os.makedirs(results_path)

plt.figure(figsize=(11, 11))
sns.scatterplot(x='MW',
                y='LogP',
                data=activity_df,
                style='bioactivity_class',
                style_order=['active','inactive','intermediate'],
                hue = 'bioactivity_class',
                hue_order=['intermediate','inactive','active'],
                edgecolor='black', alpha=0.7)
plt.ylim(-10,15)
plt.xlim(0,2000)
plt.xlabel('Molecular Weight (g/mol)', fontsize=14, fontweight='bold')
plt.ylabel('LogP (oil/water)', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.savefig(f'{results_path}/plot_MW_vs_LogP.svg',bbox_inches='tight')
    
plt.figure(figsize=(5.5, 5.5))
sns.countplot(x='bioactivity_class',
              data=activity_df,
              edgecolor='black',
              hue='bioactivity_class')
plt.xlabel('Bioactivity Class', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.savefig(f'{results_path}/freq_bioactivity.svg')

plt.figure(figsize=(5.5, 5.5))
sns.boxplot(x = 'bioactivity_class',
            y = 'MW',
            data = activity_df,
            hue='bioactivity_class',
                hue_order=['intermediate','inactive','active'])
plt.xlabel('Bioactivity Class', fontsize=14, fontweight='bold')
plt.ylabel('Molecular Weight (g/mol)', fontsize=14, fontweight='bold')
plt.savefig(f'{results_path}/MW_bioactivity.svg')

plt.figure(figsize=(5.5, 5.5))
sns.boxplot(x = 'bioactivity_class',
            y = 'LogP',
            data = activity_df,
            hue='bioactivity_class',
                hue_order=['intermediate','inactive','active'])
plt.xlabel('Bioactivity Class', fontsize=14, fontweight='bold')
plt.ylabel('LogP (oil/water)', fontsize=14, fontweight='bold')
plt.savefig(f'{results_path}/LogP_bioactivity.svg')

plt.figure(figsize=(5.5, 5.5))
sns.boxplot(x = 'bioactivity_class',
            y = 'NumHDonors',
            data = activity_df,
            hue='bioactivity_class',
                hue_order=['intermediate','inactive','active'])
plt.xlabel('Bioactivity Class', fontsize=14, fontweight='bold')
plt.ylabel('Hydrogen Donor Groups', fontsize=14, fontweight='bold')
plt.savefig(f'{results_path}/NumHDonors_bioactivity.svg')

plt.figure(figsize=(5.5, 5.5))
sns.boxplot(x = 'bioactivity_class',
            y = 'NumHAcceptors',
            data = activity_df,
            hue='bioactivity_class',
                hue_order=['intermediate','inactive','active'])
plt.xlabel('Bioactivity Class', fontsize=14, fontweight='bold')
plt.ylabel('Hydrogen Acceptor Groups', fontsize=14, fontweight='bold')
plt.savefig(f'{results_path}/NumHAcceptors_bioactivity.svg')

plt.figure(figsize=(5.5, 5.5))
cross = pd.crosstab(activity_df['Ro5Violations'],activity_df['bioactivity_class'])
rel_cross = pd.crosstab(activity_df['Ro5Violations'],activity_df['bioactivity_class']).astype('float64')
rel_cross = rel_cross.reindex(columns=['inactive','intermediate','active'])
for row in cross.index:
    for col in cross.columns:
        rel_cross.at[row,col] = (cross.at[row,col])/sum(cross[col])
sns.heatmap(data=rel_cross, annot=True)
plt.savefig(f'{results_path}/Ro5Violations_bioactivity.svg',bbox_inches='tight')

activity_df_inactive = activity_df[activity_df['bioactivity_class'] == 'inactive']
activity_df_active = activity_df[activity_df['bioactivity_class'] == 'active']
mannwhitney_results = pd.concat([
    mmm.mannwhitney_test('MW',activity_df_active,activity_df_inactive),
    mmm.mannwhitney_test('LogP',activity_df_active,activity_df_inactive),
    mmm.mannwhitney_test('NumHDonors',activity_df_active,activity_df_inactive),
    mmm.mannwhitney_test('NumHAcceptors',activity_df_active,activity_df_inactive)
])
print(mannwhitney_results)
print(f'\nPlots and Mann-Whitney U test results available at {results_path} folder\n')
mannwhitney_results.to_csv(f'{results_path}/mannwhitney_results.csv', index=False)
print('Evaluate results and adjust the plots if necessary.')
