import package_functions as pf
import pandas as pd

yeasts_df = pd.DataFrame()
data = pf.get_activity_data(target_chembl_id=["CHEMBL4513186"],activity_type="MIC")
yeasts_df = pd.concat([yeasts_df, data], ignore_index=True)
print('1')
data = pf.get_activity_data(target_chembl_id=["CHEMBL612647"],activity_type="MIC")
yeasts_df = pd.concat([yeasts_df, data], ignore_index=True)
print('2')
yeasts_df.to_csv('candidiasis_resistant.csv', index_label="index")

