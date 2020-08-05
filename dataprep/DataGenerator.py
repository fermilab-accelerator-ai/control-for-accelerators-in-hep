from keras.models import Sequential

class DataGenerator(Sequential):
    def __init__(self, file_name, variables=['B:VIMIN','B:IMINER'], backward=200, forward=30, batch_size=2, nblocks=10):

        ##
        self.batch_size = batch_size
        self.nblocks = nblocks
        self.variables = variables
        self.backward = backward
        self.forward = forward
        self.idx = 0

        ##
        self.hf = h5py.File(file_name, 'r')
        self.total_length = self.hf['ACNET/block0_values'].shape[0]
        self.nvars = self.hf['ACNET/block0_values'].shape[1]
        self.h5_variables = (hf['ACNET/block0_items'].value.astype(str))
        self.indices = []
        for var in self.variables:
            self.indices.append(np.where(h5_variables == var))
        if len(self.indices)!=len(self.variables):
            print('Not all variables are available.')

    def generate(self, debug=False, save_plots=False):

        ## Define the total length of the sample
        sample_length = self.backward + self.forward

        ## Loop over the data to make a batch
        ## TODO: Need to make a block to allow pre-fetching
        ## Shape is number of traces, number time steps, number of variables
        batch_x = np.empty(shape=(self.batch_size,self.backward,len(self.indices)))
        batch_y = np.empty(shape=(self.batch_size,self.forward,len(self.indices)))
        for b from range(self.batch_size):
            ## Check if idx is within the available data if not wrap back to the beginning
            if self.idx + sample_length > self.total_length:
                self.idx = 0
            # Define indices #
            backward_start_idx = self.idx
            backward_stop_idx = self.idx+self.backward
            forward_start_idx = backward_stop_idx+1
            forward_stop_idx = forward_start_idx + self.forward
            #for i in range(nvars):
            #    batch_x[b][] = hf['ACNET/block0_values'][backward_start_idx:backward_stop_idx, indices[i][0]]
            #    forward_trace  = hf['ACNET/block0_values'][forward_start_idx:forward_stop_idx,   indices[i][0]]
            self.idx+=1