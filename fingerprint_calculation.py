import pandas as pd
import glob
from padelpy import padeldescriptor, from_smiles
from plyer import notification
import molecules_manipulation_methods as mmm
import os
import sys

try:
    filename = sys.argv[1]
except:
    print(
        '''\nPlease insert the dataset filename as an argument, like this:
    >python exploratory_analysis.py FILENAME.csv'''
    )
    quit()

datasets_path = 'datasets'
try:
    activity_df = pd.read_csv(f'{datasets_path}/{filename}')
except:
    if not os.path.exists(f'{datasets_path}/{filename}'):
        print('\nFile does not exist')
    else:
        print('\nInvalid file - cannot convert to dataframe')
    quit()

fingerprint_files = glob.glob('fingerprint/*.xml')
fingerprint_files.sort()
fingerprint_files_df = pd.DataFrame(fingerprint_files)
for row in fingerprint_files_df.index:
    fingerprint_files_df.at[row,0] = fingerprint_files_df.at[row,0][12:-4]
print(fingerprint_files_df)
fingerprint_index:str = input('\nSelect which fingerprint method you want to use by inserting the index\n')
try:
    fingerprint_index = int(fingerprint_index)
    fingerprint = fingerprint_files[fingerprint_index]
    print(fingerprint)
except:
    print('\nPlease type only the index number of the fingerprint method you want')
    quit()

reuse = 0
if os.path.exists('descriptors.csv'):
    reuse = input('\nThere is a descriptors.csv file on the folder. Do you want to reutilize it? 1 or 0\n')
    reuse = mmm.check_if_int(reuse)

if reuse != 1:
    df_smi = activity_df['canonical_smiles']
    df_smi.to_csv('molecules.smi', sep='\t',  index=False, header=False)
    print('''\nBeginning descriptor calculation. This will create a descriptors.csv file.
It can take a couple of hours or more, depending on your dataset size and descriptors chosen.
You can check the progression at the descriptors.csv.log file that was created on this folder''')
    padeldescriptor(mol_dir='molecules.smi',
                    d_file='descriptors.csv',
                    descriptortypes=fingerprint,
                    detectaromaticity=True,
                    standardizenitro=True,
                    standardizetautomers=True,
                    threads=2,
                    removesalt=True,
                    log=True,
                    fingerprints=True)
else:
    print('\nReutilizing descriptors.csv file')

descriptors_df = pd.read_csv('descriptors.csv')
descriptors_df = descriptors_df.drop('Name',axis=1)
select_variance = input('\nType 1 to remove descriptors with low variance\n')
select_variance = mmm.check_if_int(select_variance)

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
    descriptors_df = mmm.remove_low_variance_columns(descriptors_df, variance_threshold)
    output_filename = mmm.generate_unique_filename(datasets_path,
                                                filename[:-6],
                                                f'FP{fingerprint_index}',
                                                f'VT{variance_threshold}')
else:
    output_filename = mmm.generate_unique_filename(datasets_path,
                                                filename[:-6],
                                                f'FP{fingerprint_index}')

fingerprint_df = pd.concat([
    activity_df['molecule_chembl_id'],
    descriptors_df,
    activity_df['neg_log_value'],
    activity_df['bioactivity_class']],
    axis = 1)
print('\n',fingerprint_df)
fingerprint_df.to_csv(output_filename, index=False)
print(f'\nResult is avaliable at {output_filename}')
clean = input('\nType 1 to delete temporary files\n')
clean = mmm.check_if_int(clean)

if clean == 1:   
    os.remove('molecules.smi')
    os.remove('descriptors.csv')
    os.remove('descriptors.csv.log')


