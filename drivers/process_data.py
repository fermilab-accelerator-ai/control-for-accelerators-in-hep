import dataprep.dataset as dataset

filename = '/data/fermilab-accelerator-ai/MLParamData_1583906408.4261804_From_MLrn_2020-03-10+00_00_00_to_2020-03-11+00_00_00.h5'

print(dataset.reformat_dataset(filename))
