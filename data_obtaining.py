import os
import sys
import pandas as pd
from chembl_webresource_client.new_client import new_client
import molecules_manipulation_methods as mmm
import miscelanneous_methods as mm


datasets_path = 'datasets'

if not os.path.exists(datasets_path):
    os.mkdir(datasets_path)

target_chembl_id :str = input("\nInsert ChEMBL Target ID: \n")
will_convert = input(
    """\nDo you want to convert to mol/L? 1 or 0.
Note that only nM, uM, mM and ug/mL are supported and
result in other units will be removed:\n"""
    )
will_convert = mm.check_if_int(will_convert,1)
print('\nQuerying the database, please wait')
activity = new_client.activity
activity_query = activity.filter(target_chembl_id=target_chembl_id)
activity_df : pd.DataFrame = pd.DataFrame(activity_query)

try:
    print(activity_df['standard_type'].value_counts())
except:
    print('\nThe ID might be invalid. Please note that it is case-sensitive\nExiting')
    quit()

activity_type :list[str] = input("\nChoose which type of result to obtain.\n")
print('\nFiltering for type')
activity_query = activity_query.filter(standard_type=activity_type)
activity_df : pd.DataFrame = pd.DataFrame(activity_query)
print('\nCleaning and preprocessing data')
activity_df : pd.DataFrame = pd.DataFrame(activity_query)
activity_df = activity_df[
    ['canonical_smiles', 'molecule_chembl_id', 'standard_type',
    'standard_value', 'standard_units']]
activity_df['standard_value'] = pd.to_numeric(activity_df['standard_value'], errors='coerce')
activity_df = activity_df[activity_df['standard_value'] > 0]                        
activity_df = activity_df.dropna().drop_duplicates("canonical_smiles").reset_index(drop=True)
activity_df = mmm.getLipinskiDescriptors(activity_df)
activity_df = mmm.getRo5Violations(activity_df)

if will_convert == 1:
    print('\nConverting units to mol/L')
    activity_df = mmm.convert_to_M(activity_df)

activity_df = mm.normalizeValue(activity_df)
activity_df = mm.getNegLog(activity_df)
bioactivity_class =[]

for i in activity_df.standard_value:
    if float(i) >= (0.00001): #10000 nmol/L
        bioactivity_class.append("inactive")
    elif float(i) < (0.000001): #1000 mol/L
        bioactivity_class.append("active")
    else:
        bioactivity_class.append("intermediate")

#add neg_log_value descrive() from active class, to showcase range of predictibility
        
activity_df['bioactivity_class'] = bioactivity_class
print(activity_df['bioactivity_class'].value_counts())
print(activity_df['neg_log_value'].describe())
output_filename = mm.generate_unique_filename(datasets_path, target_chembl_id[6:], activity_type)
activity_df.to_csv(output_filename, index=False)
print(activity_df)
print(f'\nResult is avaliable at {output_filename}')

