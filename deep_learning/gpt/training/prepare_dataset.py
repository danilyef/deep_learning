import tiktoken
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import argparse


def text_generator(directory):
    files = sorted(Path(directory).glob("*.txt"))
    for f in files:
        yield {
            "text": f.read_text(encoding="utf-8")
        }


def load_split(directory):
    return Dataset.from_generator(
        text_generator,
        gen_kwargs={'directory': directory}
    )

tokenizer = tiktoken.get_encoding("gpt2")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description= "GPT Data Preparation")
    parser.add_argument("--train_folder", type=str, default="/home/dan_yef/gutenberg_preprocessed/train")
    parser.add_argument("--val_folder", type=str, default="/home/dan_yef/gutenberg_preprocessed/val")
    parser.add_argument("--train_bin_file", type=str, default="/home/dan_yef/gutenberg_preprocessed/train.bin")
    parser.add_argument("--val_bin_file", type=str, default="/home/dan_yef/gutenberg_preprocessed/val.bin")
    args = parser.parse_args()


    def tokenize_datasets(example):
        ids = tokenizer.encode(example['text'], allowed_special={"<|endoftext|>"})
        out = {'ids': ids, 'len': len(ids)}
        return out
    
    dataset = DatasetDict({
        "train" : load_split(args.train_folder),
        "val": load_split(args.val_folder)
    })

    bin_files = {"train": args.train_bin_file, "val": args.val_bin_file}

    tokenized = dataset.map(
        tokenize_datasets,
        remove_columns=["text"],
        num_proc=4
    )

    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = bin_files[split]
        arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))
        total_batches = len(dset) 
        
        idx=0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids']) # to numpy array
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()

