import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

class MOSEIDataset(Dataset):
    def __init__(self, pickle_path, split, max_seqlen=50):
        """
        Args:
            pickle_path
            split: string among ["train", "val", "test"]
            max_seqlen: truncate sequence according to this parameter. Default is 50, which does no truncating

        Note that this dataset object outputs modality data under bs * seq_len * embed_dim format, note that transposing
        needs to be done for some models requiring seq_len * bs * embed_dim format. Be careful!
        """
        splits = ["train", "val", "test"]
        assert split in splits
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        modality_names = ["OpenSMILE", "OpenFace_2.0", "FACET 4.2", "glove_vectors", "COAVAREP"]
        X_data_list = [data[0][splits.index(split)][key] for key in modality_names]
        y_data = data[0][splits.index(split)]["All Labels"]
        del data # get GC cleaup this huge object
        y_data = np.squeeze(y_data)
        for i in range(len(X_data_list)):
            n, seq_len, embed_dim = X_data_list[i].shape
            nchunks = seq_len // max_seqlen
            if max_seqlen >= seq_len:
                continue
            truncated_tensor = X_data_list[i][:, :nchunks * max_seqlen, :]
            reshaped_tensor = truncated_tensor.reshape(n, nchunks, max_seqlen, embed_dim)
            X_data_list[i] = reshaped_tensor.reshape(-1, max_seqlen, embed_dim)

        self.X_list = X_data_list
        self.y = np.repeat(y_data, nchunks, axis=0)
        for X in self.X_list:
            assert X.shape[0] == self.y.shape[0]
        

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        X = [torch.Tensor(X_elem[idx]) for X_elem in self.X_list]
        y = torch.Tensor(self.y[idx])
        return *X, y

if __name__ == "__main__":
    dataset = MOSEIDataset("tensors_short.pkl", "train")
    print(len(dataset))
    for i in range(5):
        print(dataset[1][i].shape)
    
