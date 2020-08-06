import h5py
import numpy as np
from keras.models import Sequential


class DataGenerator(Sequential):
    def __init__(self, filename, variables=['B:VIMIN','B:IMINER','I:MDAT40','I:IB','B:LINFRQ'], backward=200, forward=30, batch_size=32):

        ##
        self.batch_size = batch_size
        self.variables = variables
        self.backward = backward
        self.forward = forward
        self.idx = 0

        ##
        self.hf = h5py.File(filename, 'r')
        print([key for key in self.hf.keys()])
        self.total_length = self.hf['ACNET/block0_values'].shape[0]
        self.nvars = self.hf['ACNET/block0_values'].shape[1]
        self.h5_variables = (self.hf['ACNET/block0_items'].value.astype(str))
        self.indices = []
        for var in self.variables:
            self.indices.append(np.where(self.h5_variables == var))
        if len(self.indices)!=len(self.variables):
            print('Not all variables are available.')

    def generate(self):
        while 1:
            ## Define the total length of the sample
            sample_length = self.backward + self.forward

            ## Loop over the data to make a batch
            ## TODO: Need to make a block to allow pre-fetching
            ## Shape is number of traces, number time steps, number of variables
            list_x, list_y = [],[]
            for b in range(self.batch_size):
                ## Check if idx is within the available data if not wrap back to the beginning
                if self.idx + sample_length > self.total_length:
                    self.idx = 0
                # Define indices #
                backward_start_idx = self.idx
                backward_stop_idx = self.idx+self.backward
                forward_start_idx = backward_stop_idx+1
                forward_stop_idx = forward_start_idx + self.forward
                sub_list_x,sub_list_y = [],[]
                for i in range(len(self.indices)):
                    sub_list_x.append(self.hf['ACNET/block0_values'][backward_start_idx:backward_stop_idx, self.indices[i][0]].reshape(-1))
                    sub_list_y.append(self.hf['ACNET/block0_values'][forward_start_idx:forward_stop_idx,self.indices[i][0]].reshape(-1))
                self.idx+=1
                sub_batch_x = np.stack(sub_list_x,1)
                sub_batch_y = np.stack(sub_list_y,1)
                list_x.append(sub_batch_x)
                list_y.append(sub_batch_y)

            batch_x = np.stack(list_x,0)
            batch_y = np.stack(list_y,0)
            yield batch_x,batch_y

def main():
    import h5py
    import numpy as np
    data_type = 'h5'
    filename = '../data/MLParamData_1575356421.3855522_From_MLrn_2019-12-02+00:00:00_to_2019-12-03+00:00:00.h5'
    #hf = h5py.File(filename + '_processed.{}'.format(data_type), 'r')
    training_generator = DataGenerator(filename + '_processed.{}'.format(data_type)).generate()
    batch_x, batch_y = next(training_generator)
    print(batch_x.shape)
    print(batch_y.shape)
    return 0

#if __name__ == "__main__":
#    main()