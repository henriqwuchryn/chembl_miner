import pandas as pd
import numpy as np
from chembl_webresource_client.new_client import new_client
import molecules_manipulation_methods as mmm
from plyer import notification
import os
import sys

datasets_path = 'datasets'

if not os.path.exists(datasets_path):
    os.mkdir(datasets_path)

target_chembl_id :str = input("\nInsert ChEMBL Target ID: \n")
print('\nPlease wait')
activity = new_client.activity
activity_query = activity.filter(target_chembl_id=target_chembl_id)
activity_df : pd.DataFrame = pd.DataFrame(activity_query)
# notification.notify(title='Target query done', app_name='data_obtaining')

try:
    print(activity_df['standard_type'].value_counts())
except:
    print('\nThe ID might be invalid. Please note that it is case-sensitive\nExiting')
    quit()

activity_type :list[str] = input("\nChoose which type of result to obtain.")
print('\nFiltering for type')
activity_query = activity_query.filter(standard_type=activity_type)
activity_df : pd.DataFrame = pd.DataFrame(activity_query)
# notification.notify(title='Activity type query done', app_name='data_obtaining')

# print(activity_df['standard_units'].value_counts())
# activity_units :list[str] = input(
#     """Choose which units you want to keep on the dataframe
# Type nothing if you want all. Separate with ; and a space:\n"""
#     ).split("; ")

# if activity_units != []:
#         activity_query = activity_query.filter(standard_units=activity_units)
#         print('Filtering for unit')

print('\nCleaning and preprocessing data')
activity_df : pd.DataFrame = pd.DataFrame(activity_query)
activity_df.to_csv('testando.csv')
activity_df = activity_df[['canonical_smiles',
                        'molecule_chembl_id',
                        'standard_type',                        
                        'standard_value',
                        'standard_units']]
activity_df = activity_df.dropna().drop_duplicates("canonical_smiles").reset_index(drop=True)
activity_df = mmm.getLipinskiDescriptors(activity_df)
activity_df = mmm.getRo5Violations(activity_df)
will_convert = input(
    """\nDo you want to convert to mol/L? 1 or 0.
Note that only nM, uM, mM and ug/mL are supported and
result in other units will be removed:\n"""
    )
will_convert = mmm.check_if_int(will_convert,1)

if will_convert == 1:
    print('\nConverting')
    activity_df = mmm.convert_to_M(activity_df)

activity_df = mmm.normalizeValue(activity_df)
activity_df = mmm.getNegLog(activity_df)


bioactivity_class =[]

for i in activity_df.standard_value:
    if float(i) >= (0.00001): #10000 nmol/L
        bioactivity_class.append("inactive")
    elif float(i) < (0.000001): #1000 mol/L
        bioactivity_class.append("active")
    else:
        bioactivity_class.append("intermediate")
        
activity_df['bioactivity_class'] = bioactivity_class
output_filename = mmm.generate_unique_filename(datasets_path, target_chembl_id, activity_type)
activity_df.to_csv(output_filename, index=False)
print(activity_df)
print('\n Output filename is ', output_filename)

