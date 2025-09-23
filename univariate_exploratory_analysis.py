import pandas as pd
import seaborn as sns


sns.set_theme(style='ticks')
sns.set_theme(style='whitegrid', font='liberation serif')
import matplotlib.pyplot as plt
import miscelanneous_methods as mm
import os
import sys


try:
    filename = sys.argv[1]
except:
    print(
        '''\nyou must insert the dataset filename as an argument, like this:
    >python univariate_exploratory_analysis.py FILENAME.csv''',
        )
    quit()

results_path = f'analysis/{filename[:-4]}'
datasets_path = 'datasets'

try:
    activity_df = pd.read_csv(f'{datasets_path}/{filename}', index_col='index')
except:
    if not os.path.exists(f'{datasets_path}/{filename}'):
        print('\nFile does not exist')
    else:
        print('\nInvalid file - cannot convert to dataframe')
    quit()
if not os.path.exists(results_path):
    os.makedirs(results_path)

plt.figure(figsize=(5.5, 5.5))
sns.displot(activity_df['neg_log_value'], kind='kde', bw_adjust=0.6)
plt.xlabel('Negative Log Value', fontsize=14, fontweight='bold')
plt.ylabel('Density', fontsize=14, fontweight='bold')
plt.savefig(f'{results_path}/neg_log_value_density.svg')

plt.figure(figsize=(5.5, 5.5))
cross = pd.crosstab(activity_df['Ro5Violations'], activity_df['bioactivity_class'])
rel_cross = pd.crosstab(activity_df['Ro5Violations'], activity_df['bioactivity_class']).astype('float64')
rel_cross = rel_cross.reindex(columns=['inactive', 'intermediate', 'active'])
for row in cross.index:
    for col in cross.columns:
        rel_cross.at[row, col] = (cross.at[row, col]) / sum(cross[col])
sns.heatmap(data=rel_cross, annot=True)
plt.savefig(f'{results_path}/Ro5Violations_bioactivity.svg', bbox_inches='tight')

activity_df = activity_df.drop(activity_df[activity_df['bioactivity_class'] == 'intermediate'].index)
active_df = activity_df[activity_df['bioactivity_class'] == 'active']
inactive_df = activity_df[activity_df['bioactivity_class'] == 'inactive']

plt.figure(figsize=(5.5, 5.5))
fig, ax = plt.subplots()
sns.kdeplot(active_df, x='NumHDonors', bw_adjust=0.6, label='active', ax=ax)
sns.kdeplot(inactive_df, x='NumHDonors', bw_adjust=1, label='inactive', ax=ax)
ax.set_xlim(-5, 25)
plt.xlabel('Number of Hydrogen Donor Groups', fontsize=14, fontweight='bold')
plt.ylabel('Density', fontsize=14, fontweight='bold')
plt.legend()
plt.savefig(f'{results_path}/NumHDonors_density.svg')

plt.figure(figsize=(5.5, 5.5))
fig, ax = plt.subplots()
sns.kdeplot(active_df, x='NumHAcceptors', bw_adjust=0.6, label='active', ax=ax)
sns.kdeplot(inactive_df, x='NumHAcceptors', bw_adjust=1, label='inactive', ax=ax)
ax.set_xlim(-5, 35)
plt.xlabel('Number of Hydrogen Acceptor Groups', fontsize=14, fontweight='bold')
plt.ylabel('Density', fontsize=14, fontweight='bold')
plt.legend()
plt.savefig(f'{results_path}/NumHAcceptors_density.svg')

plt.figure(figsize=(8, 8))
sns.jointplot(
    data=activity_df, x='MW', y='LogP', hue='bioactivity_class',
    hue_order=['active', 'inactive'], edgecolor='black', alpha=0.5, marginal_kws={'bw_adjust': 0.5},
    )
plt.ylim(-10, 15)
plt.xlim(0, 2000)
plt.xlabel('Molecular Weight (g/mol)', fontsize=14, fontweight='bold')
plt.ylabel('LogP (oil/water)', fontsize=14, fontweight='bold')
plt.legend()
plt.savefig(f'{results_path}/plot_MW_vs_LogP.svg', bbox_inches='tight')

plt.figure(figsize=(5.5, 5.5))
sns.countplot(
    x='bioactivity_class',
    data=activity_df,
    edgecolor='black',
    hue='bioactivity_class',
    hue_order=['active', 'inactive'],
    )
plt.xlabel('Bioactivity Class', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')
plt.savefig(f'{results_path}/freq_bioactivity.svg')

plt.figure(figsize=(5.5, 5.5))
sns.boxplot(
    x='bioactivity_class',
    y='MW',
    data=activity_df,
    hue='bioactivity_class',
    hue_order=['active', 'inactive'],
    )
plt.xlabel('Bioactivity Class', fontsize=14, fontweight='bold')
plt.ylabel('Molecular Weight (g/mol)', fontsize=14, fontweight='bold')
plt.savefig(f'{results_path}/MW_bioactivity.svg')

plt.figure(figsize=(5.5, 5.5))
sns.boxplot(
    x='bioactivity_class',
    y='LogP',
    data=activity_df,
    hue='bioactivity_class',
    hue_order=['active', 'inactive'],
    )
plt.xlabel('Bioactivity Class', fontsize=14, fontweight='bold')
plt.ylabel('LogP (oil/water)', fontsize=14, fontweight='bold')
plt.savefig(f'{results_path}/LogP_bioactivity.svg')

plt.figure(figsize=(5.5, 5.5))
sns.boxplot(
    x='bioactivity_class',
    y='NumHDonors',
    data=activity_df,
    hue='bioactivity_class',
    hue_order=['active', 'inactive'],
    )
plt.xlabel('Bioactivity Class', fontsize=14, fontweight='bold')
plt.ylabel('Hydrogen Donor Groups', fontsize=14, fontweight='bold')
plt.savefig(f'{results_path}/NumHDonors_bioactivity.svg')

plt.figure(figsize=(5.5, 5.5))
sns.boxplot(
    x='bioactivity_class',
    y='NumHAcceptors',
    data=activity_df,
    hue='bioactivity_class',
    hue_order=['active', 'inactive'],
    )
plt.xlabel('Bioactivity Class', fontsize=14, fontweight='bold')
plt.ylabel('Hydrogen Acceptor Groups', fontsize=14, fontweight='bold')
plt.savefig(f'{results_path}/NumHAcceptors_bioactivity.svg')

mannwhitney_results = pd.concat(
    [
        mm.mannwhitney_test('MW', active_df, inactive_df),
        mm.mannwhitney_test('LogP', active_df, inactive_df),
        mm.mannwhitney_test('NumHDonors', active_df, inactive_df),
        mm.mannwhitney_test('NumHAcceptors', active_df, inactive_df),
        ],
    )
print(mannwhitney_results)
print(f'\nPlots and Mann-Whitney U test results are available at {results_path} folder\n')
mannwhitney_results.to_csv(f'{results_path}/mannwhitney_results.csv', index=False)
print('Evaluate results and adjust the plots if necessary.')
