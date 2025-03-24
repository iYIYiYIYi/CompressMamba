import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from hyena.standalone_hyenadna import CharacterTokenizer
import sentencepiece as spm

import numpy as np
import random

def setup_seed(seed):
    np.random.seed(seed) 
    random.seed(seed)
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    
    #torch.use_deterministic_algorithms(warn_only=True) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False

# setup_seed(12345)
setup_seed(3407)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import deepspeed as ds
from deepspeed import DeepSpeedEngine

epochs = 300

seq_len = 128
batch_size = 768
mamba_layers = 2
hidden_dim = 64
d_model = 256
expand = 6
output_heads = 1
V_SIZE = 0

ds_stage = 0
ds_fp16 = False
ds_bf16 = True

lr = 0.001
warmup_min_lr = 0.0001
warmup_max_lr = 0.001
warmup_num_steps = 1000

model_name = 'CompressMamba'
version_name = f'-charTokenizer-bf16_{ds_bf16}-{mamba_layers}-{hidden_dim}-{d_model}-{expand}-{output_heads}-v{V_SIZE}'

from models import get_CompressMamba, get_DeepDNA_Attn, get_DeepDNA, get_DeepDNA_DB, get_CompressAttn

model_name_dict = {
    'CompressMamba': get_CompressMamba,
    'DeepDNA_Attn': get_DeepDNA_Attn,
    'DeepDNA': get_DeepDNA,
    'DeepDNA_DB': get_DeepDNA_DB,
    'CompressAttn': get_CompressAttn,
}

model_arg_dict = {
    'CompressMamba': [seq_len, V_SIZE, 'single', hidden_dim, mamba_layers, expand, d_model, output_heads],
    'DeepDNA_Attn':  [seq_len],
    'DeepDNA':       [seq_len],
    'DeepDNA_DB':    [seq_len],
    'CompressAttn':  [seq_len, V_SIZE],
}

import json

model, tokenizer, tokenizer_args = None, None, None
compress_data = None
SPECIES_STATISTICS = None
FILENAME_STATISTICS = None
def init_models():
    global model, tokenizer, tokenizer_args, compress_data, SPECIES_STATISTICS, FILENAME_STATISTICS

    if V_SIZE != 0:
        model_arg_dict['CompressMamba'] = [seq_len, V_SIZE, 'sentencepiece', hidden_dim, mamba_layers, expand, d_model, output_heads]

    model, tokenizer, tokenizer_args = model_name_dict[model_name](*model_arg_dict[model_name])
    compress_data = CompressCSVData(batch_size=batch_size, seq_len=seq_len, output_heads=output_heads, token_version=V_SIZE, tokenizer=tokenizer, tokenizer_args=tokenizer_args)

    SPECIES_STATISTICS = {
        'train': compress_data.get_train_species_statistics(),
        'valid': compress_data.get_valid_species_statistics(),
        'test': compress_data.get_test_species_statistics(),
    }

    FILENAME_STATISTICS = {
        'train': compress_data.get_train_filename_statistics(),
        'valid': compress_data.get_valid_filename_statistics(),
        'test': compress_data.get_test_filename_statistics(),
    }

    args = {
        'seq_len': seq_len,
        'batch_size': batch_size,
        'mamba_layers': mamba_layers,
        'hidden_dim': hidden_dim,
        'd_model': d_model,
        'expand': expand,
        'output_heads': output_heads,
        'version_name': version_name,
        'model_name': model_name,
        'ds_stage': ds_stage,
        'ds_fp16': ds_fp16,
        'ds_bf16': ds_bf16,
    }

    with open(f'./model_args/{model_name}{version_name}.json', 'w') as f:
        json.dump(args, f)


def get_ds_config(stage=2, fp16=False, bf16=False):
    """Get the DeepSpeed configuration dictionary."""
    ds_config = {
        "train_batch_size": batch_size,
        "steps_per_print": 1000,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": lr,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": warmup_min_lr,
                "warmup_max_lr": warmup_max_lr,
                "warmup_num_steps": warmup_num_steps,
            },
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "bf16": {
            "enabled": bf16,
            "auto_mixed_precision": True,
        },
        "fp16": {
            "enabled": fp16,
            "auto_cast": True,
            "loss_scale": 0,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "consecutive_hysteresis": False,
            "min_loss_scale": 1
        },
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": stage,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 50000000,
            "reduce_bucket_size": 50000000,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "cpu_offload": False,
        },
        # "flops_profiler": {
        #     "enabled": True,
        #     "profile_step": 1,
        #     "module_depth": -1,
        #     "top_modules": 1,
        #     "detailed": True,
        #     "output_file": None,
        # }
    }
    return ds_config


from utils.compress_csv_dataset import CompressCSVData, transfer_file_into_index, transfer_species_into_index, transfer_index_into_file, transfer_index_into_species
from standalone_compressmamba import CompressMamba
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from utils.species import SPECIES, ALL_SPECIES
import tqdm
import csv
from thop import profile

def summary(model):
    v_size = V_SIZE
    if tokenizer.__class__.__name__ == 'SentencePieceProcessor':
        v_size = tokenizer.get_piece_size()
    else:
        v_size = tokenizer.vocab_size
    model = model.cuda()
    input = torch.randint(0, v_size, (1, seq_len)).cuda()  # random input
    flops, params = profile(model, inputs=(input,))
    print("="*42+"-Model Summary-"+"="*43)
    print(model)
    print("=" * 100)
    # calc model's FLOPs and params
    print("Total Trainable Parameters: ", params)
    print("=" * 100)
    print("Total FLOPs: ", flops)
    print("=" * 100)
    print("Flops: ", flops / 1e9, "G")
    print("=" * 100)
    model = model.cpu()
    torch.cuda.empty_cache()

class ModelTrain:
    def __init__(self, tokenizer, model, train_callbacks=None, val_callbacks=None, test_callbacks=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        summary(self.model)
        self.model_engine, self.optimizer, _, self.lr_scheduler = ds.initialize(
            model=self.model,
            # model_parameters=[p for p in self.model.parameters() if p.requires_grad],
            config=get_ds_config(ds_stage, ds_fp16, ds_bf16)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.N = 0
        self.softmax = nn.Softmax(dim=-1)

        self.train_callbacks = train_callbacks
        self.val_callbacks = val_callbacks
        self.test_callbacks = test_callbacks

        self.test_avg_entropy = {}
        self.test_avg_original_entropy = {}

        self.test_avg_entropy_file = {}
        self.test_avg_original_entropy_file = {}
        self.last_loss = 0
        self.early_stopping_steps = 10
        self.stop_training = False
        self.early_stopping_counter = 0

        self.head_weights = [10, 4, 2, 2, 2]
        self.head_weights = np.array(self.head_weights[0:output_heads], dtype=np.float32)
        self.head_weights /= np.sum(self.head_weights)

        self.init_entropies()

    def init_entropies(self):
        for specie in ALL_SPECIES:
            self.test_avg_entropy[specie] = 0
            self.test_avg_original_entropy[specie] = 0

        for filename in SPECIES.keys():
            self.test_avg_entropy_file[filename] = 0
            self.test_avg_original_entropy_file[filename] = 0

    def run_step(self, idx, batch, train=False, callbacks=None):
        self.model_engine.zero_grad()
        x, y, global_segment, specie, filename = batch
        x = x.to(self.model_engine.device)
        y = y.to(self.model_engine.device)
        global_segment = global_segment.to(self.model_engine.device)
        output = self.model_engine(x, global_segment)
        total_loss = 0
        y_cnt = 0
        for output_per_head in output:
            output_per_head = output_per_head.view(-1, output_per_head.size(-1))
            y_per_head = y[:, y_cnt].view(-1)
            loss = self.criterion(output_per_head, y_per_head)
            total_loss += loss * self.head_weights[y_cnt]
                
            y_cnt += 1
            if torch.isnan(total_loss):
                print('nan detected')
                exit(-1)

        self.model_engine.backward(total_loss)
        self.model_engine.step()

        if callbacks is not None:
            for cb in callbacks:
                cb(self.model_engine, batch, idx)

        return total_loss.item() / len(output)
    
    def calc_entropy(self, x, original_base_lengths=1):
        # x = x.view(x.size(-1))
        x = self.softmax(x)
        x = x * torch.log(x + 1e-9)
        x = x.sum(dim=-1)
        original_en = x / original_base_lengths
        original_en = original_en.masked_fill(torch.isnan(original_en), original_en[~torch.isnan(original_en)].mean())
        # original_en = original_en.masked_fill(torch.isnan(original_en), 0)
        original_en = -1.0 * original_en
        x = -1.0 * x
        return x, original_en
    
    
    def run_test_step(self, idx, batch, callbacks=None):
        x, y, global_segment, specie, filename = batch
        x = x.to(self.model_engine.device)
        y = y.to(self.model_engine.device)
        global_segment = global_segment.to(self.model_engine.device)
        output = self.model_engine(x, global_segment)
        total_entropy = 0
        total_original_entropy = 0
        y_cnt = 0
        for output_per_head in output:
            output_per_head = output_per_head.view(-1, output_per_head.size(-1))
            # output_probs = self.softmax(output_per_head)
            y_sub = y[:, y_cnt]
            y_cnt += 1
            # 将y转换为one_hot编码
            if tokenizer.__class__.__name__ == 'SentencePieceProcessor':
                y_sub_l = y_sub.tolist()
                original_base = [tokenizer.IdToPiece(id) for id in y_sub_l]
            else:
                original_base = tokenizer.convert_ids_to_tokens(y_sub)
            original_base_lengths = torch.tensor([len(base) for base in original_base])
            original_base_lengths = original_base_lengths.to(self.model_engine.device)
            y_sub = F.one_hot(y_sub, num_classes=output_per_head.size(-1))
            # output_per_head = output_per_head[y_sub == 1]
            entropy, original_entropy = self.calc_entropy(output_per_head, original_base_lengths)
            total_entropy += entropy.sum() / entropy.size(0)
            total_original_entropy += original_entropy.sum() / original_entropy.size(0)

            for i, sp in enumerate(specie):
                self.test_avg_entropy[transfer_index_into_species(sp)] += entropy[i].item()
                self.test_avg_original_entropy[transfer_index_into_species(sp)] += original_entropy[i].item()
                self.test_avg_entropy_file[transfer_index_into_file(filename[i])] += entropy[i].item()
                self.test_avg_original_entropy_file[transfer_index_into_file(filename[i])] += original_entropy[i].item()


        if callbacks is not None:
            for cb in callbacks:
                cb(self.model_engine, batch, idx)

        return total_entropy.item() / len(output), total_original_entropy.item() / len(output)

    def training_step(self, epoch, dataloader):
        total_loss = 0
        self.model_engine.train()
        for idx, batch in enumerate(dataloader):
            total_loss += self.run_step(idx, batch, train=True, callbacks=self.train_callbacks)

        total_loss /= len(dataloader)
        print(f'[{epoch}/{epochs}]train loss: {total_loss}')

        # early stopping
        if self.last_loss != 0 and total_loss > self.last_loss:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_steps:
                self.stop_training = True
                print(f'Early stopping at epoch {epoch}')
        else:
            self.early_stopping_counter = 0
        self.last_loss = total_loss


    def validation_step(self, epoch, dataloader):
        self.model_engine.eval()
        total_entropy = 0
        total_original_entropy = 0
        self.init_entropies()
        for idx, batch in enumerate(dataloader):
            entropies = self.run_test_step(idx, batch, callbacks=self.val_callbacks)
            total_entropy += entropies[0]
            total_original_entropy += entropies[1]

        total_entropy /= len(dataloader)
        total_original_entropy /= len(dataloader)
        print(f'[{epoch}]valid entropy: {total_entropy}')
        print(f'[{epoch}]valid original entropy: {total_original_entropy}')
        # print entropies as a table in std
        print(f'{"Species":<20}{"Entropy":<20}{"Original Entropy":<20}')
        for specie in ALL_SPECIES:
            self.test_avg_entropy[specie] /= SPECIES_STATISTICS['valid'][specie] if SPECIES_STATISTICS['valid'][specie] != 0 else 1
            self.test_avg_entropy[specie] /= output_heads
            self.test_avg_original_entropy[specie] /= SPECIES_STATISTICS['valid'][specie] if SPECIES_STATISTICS['valid'][specie] != 0 else 1
            self.test_avg_original_entropy[specie] /= output_heads
            print(f'{specie:<20}{self.test_avg_entropy[specie]:<20}{self.test_avg_original_entropy[specie]:<20}')
        print(f'{"Filename":<20}{"Entropy":<20}{"Original Entropy":<20}')
        for filename in SPECIES.keys():
            self.test_avg_entropy_file[filename] /= FILENAME_STATISTICS['valid'][filename] if FILENAME_STATISTICS['valid'][filename] != 0 else 1
            self.test_avg_entropy_file[filename] /= output_heads
            self.test_avg_original_entropy_file[filename] /= FILENAME_STATISTICS['valid'][filename] if FILENAME_STATISTICS['valid'][filename] != 0 else 1
            self.test_avg_original_entropy_file[filename] /= output_heads
            print(f'{filename:<20}{self.test_avg_entropy_file[filename]:<20}{self.test_avg_original_entropy_file[filename]:<20}')


    def test_step(self, epoch, dataloader):
        total_entropy = 0
        total_original_entropy = 0
        self.model_engine.eval()
        self.init_entropies()
        for idx, batch in enumerate(dataloader):
            entropies = self.run_test_step(idx, batch, callbacks=self.test_callbacks)
            total_entropy += entropies[0]
            total_original_entropy += entropies[1]

        total_entropy /= len(dataloader)
        total_original_entropy /= len(dataloader)
        print(f'[{epoch}]test entropy: {total_entropy}')
        print(f'[{epoch}]test original entropy: {total_original_entropy}')
        csv_file = open(f'./output/{model_name}{version_name}_entropy.csv', 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['timestamp', 'name', 'epoch', 'entropy', 'original_entropy'])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_writer.writerow([timestamp, 'total', epoch, total_entropy, total_original_entropy])
        for specie in ALL_SPECIES:
            self.test_avg_entropy[specie] /= SPECIES_STATISTICS['test'][specie] if SPECIES_STATISTICS['test'][specie] != 0 else 1
            self.test_avg_entropy[specie] /= output_heads
            self.test_avg_original_entropy[specie] /= SPECIES_STATISTICS['test'][specie] if SPECIES_STATISTICS['test'][specie] != 0 else 1
            self.test_avg_original_entropy[specie] /= output_heads
            csv_writer.writerow([timestamp, specie, epoch, self.test_avg_entropy[specie], self.test_avg_original_entropy[specie]])
        for filename in SPECIES.keys():
            self.test_avg_entropy_file[filename] /= FILENAME_STATISTICS['test'][filename] if FILENAME_STATISTICS['test'][filename] != 0 else 1
            self.test_avg_entropy_file[filename] /= output_heads
            self.test_avg_original_entropy_file[filename] /= FILENAME_STATISTICS['test'][filename] if FILENAME_STATISTICS['test'][filename] != 0 else 1
            self.test_avg_original_entropy_file[filename] /= output_heads
            csv_writer.writerow([timestamp, filename, epoch, self.test_avg_entropy_file[filename], self.test_avg_original_entropy_file[filename]])
        csv_file.close()

from datetime import datetime

import re

def main():
    checkpoint_path = './checkpoints/' + model_name + version_name
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)

    ds.init_distributed()
    print(f'Start training Model {model_name}')
    model_train = ModelTrain(tokenizer, model,
                                     train_callbacks=[], 
                                     val_callbacks=[], 
                                     test_callbacks=[])
    train_dataloader = compress_data.train_dataloader()
    val_dataloader = compress_data.val_dataloader()
    test_dataloader = compress_data.test_dataloader()

    # 从checkpoint恢复模型
    # 根据checkpoint_path下最新的模型恢复
    # 遍历checkpoint_path文件夹获取最新模型
    start_epoch = 0
    # for folder in os.listdir(checkpoint_path):
    #     match = re.search(r'model-(\d+)\.pt', folder)
    #     if match:
    #         epoch = int(match.group(1))
    #         if epoch >= start_epoch:
    #             start_epoch = epoch+1
    # if start_epoch > 0:
    #     model_train.model_engine.load_checkpoint(f"{checkpoint_path}/model-{start_epoch-1}.pt")

    for epoch in range(start_epoch, epochs):
        model_train.training_step(epoch, train_dataloader)
        if epoch % 5 == 0:
            model_train.validation_step(epoch, val_dataloader)
        if model_train.stop_training:
            break
    model_train.model_engine.save_checkpoint(f"{checkpoint_path}/model-{epoch+1}.pt")
    print(f'Finished training Model {model_name}')
    model_train.test_step(epoch, test_dataloader)


import sys
if __name__ == '__main__':
    args = sys.argv
    if len(args) > 2 and args[2] in model_name_dict.keys():
        model_name = args[2]
    if len(args) > 3 and str.isdigit(args[3]):
        V_SIZE = int(args[3])
        if V_SIZE != 0:
            version_name = f'-sentencepieceTokenizer-bf16_{ds_bf16}-{mamba_layers}-{hidden_dim}-{d_model}-{expand}-{output_heads}-v{V_SIZE}'
    if len(args) > 4 and str.isdigit(args[4]):
        output_heads = int(args[4])
        if output_heads != 1:
            version_name = f'-sentencepieceTokenizer-bf16_{ds_bf16}-{mamba_layers}-{hidden_dim}-{d_model}-{expand}-{output_heads}-v{V_SIZE}'

    print(f'Using Model {model_name} with V_SIZE {V_SIZE}')
    init_models()
    main()