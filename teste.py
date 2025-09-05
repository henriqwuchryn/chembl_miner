import package_functions as pf
import pandas as pd

teste = pf.get_activity_data(target_chembl_id="CHEMBL387",activity_type="IC50")
print(teste)
