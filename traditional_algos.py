import gzip, zstandard, bz2

def compress_seq(seq, method='zstd'):
    if method == 'zstd':
        return zstandard.compress(seq.encode())
    elif method == 'gzip':
        return gzip.compress(seq.encode())
    elif method == 'bz2':
        return bz2.compress(seq.encode())
    else:
        raise ValueError("method should be in ['zstd', 'gzip', 'bz2']")


suffix = '_min.csv'
DATA_ROOT_PATH = '/home/huangyi/data_m/compress_dataset/'
test_file = DATA_ROOT_PATH + 'test' + suffix

import csv

output_file = open(f'./output/tradition_algos_entropy.csv', 'w', newline='')
writer = csv.writer(output_file)
writer.writerow(['timestamp', 'compressor', 'name', 'epoch', 'entropy', 'original_len', 'comp_ratio'])
test_avg_species_entropy = {}
test_avg_file_entropy = {}

from utils.species import SPECIES, SPECIE_FILE_INDEX, ALL_SPECIES

for species in ALL_SPECIES:
    test_avg_species_entropy[species] = 0, 0, 0
for file in SPECIE_FILE_INDEX:
    test_avg_file_entropy[file] = 0, 0, 0

all_original_len_species = {}
all_original_len_file = {}

for species in ALL_SPECIES:
    all_original_len_species[species] = 0
for file in SPECIE_FILE_INDEX:
    all_original_len_file[file] = 0

comp_ratio = {}
for species in ALL_SPECIES:
    comp_ratio[species] = 0
for file in SPECIE_FILE_INDEX:
    comp_ratio[file] = 0

with open(test_file, 'r') as f:
    dict_reader = csv.DictReader(f)
    for row in dict_reader:
        seq = row['sequence']
        species = row['species']
        file = row['file']
        zstd_seq = compress_seq(seq, method='zstd')
        gzip_seq = compress_seq(seq, method='gzip')
        bz2_seq = compress_seq(seq, method='bz2')
        test_avg_species_entropy[species] = (test_avg_species_entropy[species][0] + len(zstd_seq), 
                                            test_avg_species_entropy[species][1] + len(gzip_seq), 
                                            test_avg_species_entropy[species][2] + len(bz2_seq))
        
        test_avg_file_entropy[file] = (test_avg_file_entropy[file][0] + len(zstd_seq), 
                                       test_avg_file_entropy[file][1] + len(gzip_seq), 
                                       test_avg_file_entropy[file][2] + len(bz2_seq))

        all_original_len_species[species] += len(seq)
        all_original_len_file[file] += len(seq)

for species in ALL_SPECIES:
    comp_ratio[species] = all_original_len_species[species] / (test_avg_species_entropy[species][0] if test_avg_species_entropy[species][0] != 0 else 1), \
                          all_original_len_species[species] / (test_avg_species_entropy[species][1] if test_avg_species_entropy[species][1] != 0 else 1), \
                          all_original_len_species[species] / (test_avg_species_entropy[species][2] if test_avg_species_entropy[species][2] != 0 else 1)
    print(f'{species} zstd: {comp_ratio[species][0]}, gzip: {comp_ratio[species][1]}, bz2: {comp_ratio[species][2]}')
for file in SPECIE_FILE_INDEX:
    comp_ratio[file] = all_original_len_file[file] / (test_avg_file_entropy[file][0] if test_avg_file_entropy[file][0] != 0 else 1), \
                       all_original_len_file[file] / (test_avg_file_entropy[file][1] if test_avg_file_entropy[file][1] != 0 else 1), \
                       all_original_len_file[file] / (test_avg_file_entropy[file][2] if test_avg_file_entropy[file][2] != 0 else 1)

for species in ALL_SPECIES:
    test_avg_species_entropy[species] = (test_avg_species_entropy[species][0] * 8 / (all_original_len_species[species] if all_original_len_species[species] != 0 else 1), 
                                        test_avg_species_entropy[species][1] * 8 / (all_original_len_species[species] if all_original_len_species[species] != 0 else 1), 
                                        test_avg_species_entropy[species][2] * 8 / (all_original_len_species[species] if all_original_len_species[species] != 0 else 1))
    print(f'{species} zstd: {test_avg_species_entropy[species][0]}, gzip: {test_avg_species_entropy[species][1]}, bz2: {test_avg_species_entropy[species][2]}')
    
for file in SPECIE_FILE_INDEX:
    test_avg_file_entropy[file] = (test_avg_file_entropy[file][0] * 8 / (all_original_len_file[file] if all_original_len_file[file] != 0 else 1), 
                                   test_avg_file_entropy[file][1] * 8 / (all_original_len_file[file] if all_original_len_file[file] != 0 else 1), 
                                   test_avg_file_entropy[file][2] * 8 / (all_original_len_file[file] if all_original_len_file[file] != 0 else 1))

import datetime
timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
for species in ALL_SPECIES:
    writer.writerow([timestamp, 'zstd', species, 0, test_avg_species_entropy[species][0], all_original_len_species[species], comp_ratio[species][0]])
    writer.writerow([timestamp, 'gzip', species, 0, test_avg_species_entropy[species][1], all_original_len_species[species], comp_ratio[species][1]])
    writer.writerow([timestamp, 'bz2' , species, 0, test_avg_species_entropy[species][2], all_original_len_species[species], comp_ratio[species][2]])
        
for file in SPECIE_FILE_INDEX:
    writer.writerow([timestamp, 'zstd', file, 0, test_avg_file_entropy[file][0], all_original_len_file[file], comp_ratio[file][0]])
    writer.writerow([timestamp, 'gzip', file, 0, test_avg_file_entropy[file][1], all_original_len_file[file], comp_ratio[file][1]])
    writer.writerow([timestamp, 'bz2' , file, 0, test_avg_file_entropy[file][2], all_original_len_file[file], comp_ratio[file][2]])

