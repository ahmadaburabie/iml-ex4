import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np


class EuropeDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
        """
        # Load the data
        self.data_frame = pd.read_csv(csv_file)
        
        # Assuming the first two columns are the features and the third is the label
        self.features = torch.from_numpy(self.data_frame.iloc[:, 1:3].values).float()
        self.labels = torch.from_numpy(self.data_frame.iloc[:, 3].values).long()

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data_frame)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data row
        
        Returns:
            dictionary: Feature tensor and corresponding label tensor
        """
        # Fetch features and label
        features = self.features[idx]
        label = self.labels[idx]  # Assuming label is an integer class label        
        sample = [features, label]
        return sample




class ShakespeareDataset(Dataset):
     
    def __init__(self, text, block_size, encoder):
        self.block_size = block_size    
        self.text = text    
        self.data = np.array(encoder(self.text))

    

    def __len__(self):
        return len(self.data)-self.block_size 
        


    def __getitem__(self, idx):
        
        x = self.data[idx:idx+self.block_size]        
        y = self.data[idx+1:idx+1+self.block_size]
                
        return x, y



          
class DataHandler:
    def __init__(self, train_filename, test_filename=None, block_size=10):

        
        with open(train_filename, 'r', encoding='utf-8') as file:
            train_text = file.read()
        
        try:
            with open(test_filename, 'r', encoding='utf-8') as file:
                test_text = file.read()
        except OSError as e:
            print(f'Error reading {test_filename}, data has only train')

        self.block_size = block_size
        self.vocab = sorted(list(set(train_text)))
        print("Vocabulary:, ", "".join(self.vocab))
        print("Length of vocabulary: ", len(self.vocab))


            

        # create a mapping from characters to integers
        stoi = {ch: i for i, ch in enumerate(self.vocab)}
        itos = {i: ch for i, ch in enumerate(self.vocab)}
        self.encoder = lambda s: [
            stoi[c] for c in s
        ]  # encoder: take a string, output a list of integers
        self.decoder = lambda l: "".join(
            [itos[i] for i in l]
        )  # decoder: take a list of integers, output a string
        
        
        self.train_dataset = ShakespeareDataset(train_text, self.block_size, self.encoder)
        if test_filename is not None:
            self.test_dataset = ShakespeareDataset(test_text, self.block_size, self.encoder)
        else: 
            self.test_dataset = None
        
        
    
    def get_vocab_size(self):
        return len(self.vocab)
    
    def get_dataset(self, mode='train'):
        if mode=='train':
            return self.train_dataset
        elif mode=='test' and self.test_dataset:
            return self.test_dataset
        print(f"Counltn't locate {mode} dataset") 
        return 
