from virny.datasets import ACSEmploymentDataset

data_loader = ACSEmploymentDataset(state=['CA'], year=2018, with_nulls=False)
print('data_loader.full_df.shape:', data_loader.full_df.shape)
