import package_functions as pf

data = pf.get_activity_data(target_chembl_id="CHEMBL366",activity_type="MIC")
print(pf.review_assays(data,assay_keywords=['ATCC'],n_to_show=50))


