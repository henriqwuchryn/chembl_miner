import glob
import os
import sys

import numpy as np
import pandas as pd
import sklearn.preprocessing as preproc

import machine_learning_methods as mlm
import miscelanneous_methods as mm
import molecules_manipulation_methods as mmm


try:
    filename = sys.argv[1]
except:
    print(
        '''\nPlease insert the dataset filename as an argument, like this:
    >python exploratory_analysis.py FILENAME.csv''',
        )
    quit()

datasets_path = 'datasets'
try:
    activity_df = pd.read_csv(f'{datasets_path}/{filename}', index_col='index')
except:
    if not os.path.exists(f'{datasets_path}/{filename}'):
        print('\nFile does not exist')
    else:
        print('\nInvalid file - cannot convert to dataframe')
    quit()

if not os.path.exists(datasets_path):
    os.makedirs(datasets_path)
fingerprint_files = glob.glob('fingerprint/*.xml')
fingerprint_files.sort()
fingerprint_files_df = pd.DataFrame(fingerprint_files)
for row in fingerprint_files_df.index:
    fingerprint_files_df.at[row, 0] = fingerprint_files_df.at[row, 0][12:-4]
print(fingerprint_files_df)
fingerprint_index: str = input(
    '\nSelect which fingerprint method you want to use by inserting the index\nNegative index will use all fingerprints\n',
    )
try:
    fingerprint_index = int(fingerprint_index)
    if fingerprint_index >= 0:
        fingerprint = fingerprint_files[fingerprint_index]
except:
    print('\nPlease type only the index number of the fingerprint method you want')
    quit()

reuse = 0
if os.path.exists('descriptors.csv'):
    reuse = input('\nThere is a descriptors.csv file on the folder. Do you want to reutilize it? 1 or 0\n')
    reuse = mm.check_if_int(reuse)

if reuse != 1:
    if fingerprint_index >= 0:
        mmm.calculate_fingerprint(activity_df, fingerprint)
    if fingerprint_index < 0:
        descriptors_df = pd.DataFrame(index=activity_df.index)
        for i in fingerprint_files:
            mmm.calculate_fingerprint(activity_df, i)
            descriptors_df_i = pd.read_csv('descriptors.csv')
            descriptors_df_i = descriptors_df_i.drop('Name', axis=1)
            descriptors_df_i = pd.DataFrame(descriptors_df_i, index=activity_df.index)
            descriptors_df = pd.concat([descriptors_df, descriptors_df_i], axis=1)
            os.remove('descriptors.csv')
            os.remove('descriptors.csv.log')
        descriptors_df.to_csv('descriptors.csv')
else:
    print('\nReutilizing descriptors.csv file')
    fingerprint_index = int(open('fingerprint_used', 'r').read())

descriptors_df = pd.read_csv('descriptors.csv')
if fingerprint_index >= 0:
    descriptors_df = descriptors_df.drop('Name', axis=1)
descriptors_df = pd.DataFrame(descriptors_df, index=activity_df.index)
descriptors_df = mlm.scale_features(descriptors_df, preproc.MinMaxScaler())

select_variance = input('\nType 1 to remove descriptors with low variance\n')
select_variance = mm.check_if_int(select_variance)

if select_variance == 1:
    variance_threshold = input('\nType a value for Variance Treshold, between 0 and 1. Default is 0.1\n')
    try:
        variance_threshold = float(variance_threshold)
        if not 0 <= variance_threshold <= 1:
            print('\nValue must be between 0 and 1. Using default value of 0.1')
            variance_threshold = 0.1
    except:
        print('\nNot a number. Using default value of 0.1')
        variance_threshold = 0.1
    descriptors_df = mm.remove_low_variance_columns(descriptors_df, variance_threshold)
    output_filename = mm.generate_unique_filename(
        datasets_path, filename[:-4],
        f'fp{fingerprint_index}', f'vt{variance_threshold}',
        )
else:
    output_filename = mm.generate_unique_filename(
        datasets_path, filename[:-4], f'fp{fingerprint_index}',
        )

fingerprint_df = pd.concat(
    [
        activity_df,
        descriptors_df,
        ],
    axis=1,
    )
fingerprint_df.replace([np.inf, -np.inf], np.nan, inplace=True)
fingerprint_df.dropna(inplace=True)
print('\n', fingerprint_df)
fingerprint_df.to_csv(output_filename, index=True, index_label='index')
print(f'\nResult is avaliable at {output_filename}')
clean = input('\nType 1 to delete temporary files\nYou can keep them to reutilize descriptors.csv\n')
clean = mm.check_if_int(clean)

if clean == 1:
    os.remove('molecules.smi')
    os.remove('descriptors.csv')
    os.remove('descriptors.csv.log')
    os.remove('fingerprint_used')
else:
    with open('fingerprint_used', 'w') as f:
        f.write(str(fingerprint_index))
        f.close()
    print('\nTemporary files kept for reutilization')
