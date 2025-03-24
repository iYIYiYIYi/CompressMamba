DATA_ROOT_PATH = '/home/huangyi/data_m/compress_dataset/'

import os
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import csv
from utils.species import SPECIES, SPECIE_FILE_INDEX, ALL_SPECIES

CSV_HEADER = ['file', 'species', 'global_position', 'sequence', 'tokens_128', 'tokens_256', 'tokens_512', 'tokens_768', 'tokens_1024', 'tokens_2048', 'tokens_4096', 'tokens_8192']
def transfer_file_into_index(file):
    return SPECIE_FILE_INDEX.index(file)

def transfer_index_into_file(index):
    return SPECIE_FILE_INDEX[index]

def transfer_species_into_index(species):
    return ALL_SPECIES.index(species)

def transfer_index_into_species(index):
    return ALL_SPECIES[index]



class CompressSeqDataset(Dataset):
    def __init__(self, seq_len, output_heads=1, mode='train', token_version=0, tokenizer=None):
        super().__init__()
        self.seq_len = seq_len
        self.mode = mode
        self.output_heads = output_heads
        self.token_version = token_version
        self.tokenizer = tokenizer
        if token_version != 0 and token_version in [128, 256, 512, 768, 1024, 2048, 4096, 8192]:
            self.token_row = f'tokens_{token_version}'
        else:
            self.token_row = 'sequence'

        self.SPECIES_STATISTICS = {}
        for species in SPECIES.values():
            self.SPECIES_STATISTICS[species] = 0

        self.FILENAME_STATISTICS = {}
        for file in SPECIES.keys():
            self.FILENAME_STATISTICS[file] = 0

        suffix = '_min.csv'
        # suffix = '.csv'

        if mode == 'train':
            self.data_path = DATA_ROOT_PATH + 'train' + suffix
        elif mode == 'valid':
            self.data_path = DATA_ROOT_PATH + 'valid' + suffix
        elif mode == 'test':
            self.data_path = DATA_ROOT_PATH + 'test' + suffix
        else:
            raise ValueError("mode type wrong, mode should be in [train, test, valid]")
        
        self.data = []
        self.species = []
        self.global_position = []
        self.file = []
        with open(self.data_path, 'r') as f:
            dict_reader = csv.DictReader(f)
            for row in dict_reader:
                # first we turn the str sequence into a list of integers
                tkn_list = []
                if token_version == 0 and tokenizer is not None:
                    # if tokenizer is spm, we need to use the EncodeAsIds method, otherwise we use the tokenizer method
                    if tokenizer.__class__.__name__ == 'SentencePieceProcessor':
                        tkn_list = tokenizer.EncodeAsIds(row['sequence'])
                    else:
                        tkn_list = tokenizer(row['sequence'])['input_ids']
                else:
                    tkn_str = row[self.token_row]
                    tkn_list = list(map(int, tkn_str.split(',')))
                self.data.append(torch.tensor(tkn_list))
                self.species.append(transfer_species_into_index(row['species']))
                self.global_position.append(int(row['global_position']))
                self.file.append(transfer_file_into_index(row['file']))
                self.SPECIES_STATISTICS[row['species']] += 1
                self.FILENAME_STATISTICS[row['file']] += 1

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx][0: self.seq_len]
        y = self.data[idx][self.seq_len: self.seq_len + self.output_heads]
        return x, y, self.global_position[idx], self.species[idx], self.file[idx]
    
    def get_species_statistics(self):
        return self.SPECIES_STATISTICS
    
    def get_filename_statistics(self):
        return self.FILENAME_STATISTICS


            
class CompressCSVData:
    def __init__(self, batch_size, seq_len, token_version, output_heads, tokenizer=None, tokenizer_args=None):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.output_heads = output_heads
        self.tokenizer_args = tokenizer_args
        self.num_workers = 0
        self.pin_memory = False

        self.train = CompressSeqDataset(seq_len=self.seq_len, mode='train', output_heads=self.output_heads, token_version=token_version, tokenizer=self.tokenizer)
        self.valid = CompressSeqDataset(seq_len=self.seq_len, mode='valid', output_heads=self.output_heads, token_version=token_version, tokenizer=self.tokenizer)
        self.test  = CompressSeqDataset(seq_len=self.seq_len, mode='test' , output_heads=self.output_heads, token_version=token_version, tokenizer=self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)
    
    def get_train_species_statistics(self):
        return self.train.get_species_statistics()
    
    def get_train_filename_statistics(self):
        return self.train.get_filename_statistics()
    
    def get_valid_species_statistics(self):
        return self.valid.get_species_statistics()
    
    def get_valid_filename_statistics(self):
        return self.valid.get_filename_statistics()
    
    def get_test_species_statistics(self):
        return self.test.get_species_statistics()
    
    def get_test_filename_statistics(self):
        return self.test.get_filename_statistics()

