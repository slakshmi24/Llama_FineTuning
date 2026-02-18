
"""
Data Preparation Script for Llama Fine-Tuning
Handles data loading, preprocessing, and tokenization
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from tqdm import tqdm


class DataProcessor:
    """Process and tokenize data for fine-tuning"""
    
    def __init__(self, model_name: str = "gpt2", max_seq_length: int = 512):
        # Use gpt2 as default for demo (no auth required)
        # For Llama models, login first: huggingface-cli login
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_seq_length = max_seq_length
    
    def format_instruction(self, example: Dict) -> str:
        """Format instruction into a single string"""
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        return prompt + output + self.tokenizer.eos_token
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """Tokenize examples"""
        texts = [self.format_instruction({"instruction": instr, 
                                           "input": inp, 
                                           "output": out})
                for instr, inp, out in zip(examples["instruction"], 
                                          examples["input"], 
                                          examples["output"])]
        
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors=None
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    def load_and_process(self, input_file: str, output_file: str, 
                        train_split: float = 0.8, val_split: float = 0.1):
        """Load JSON data, process, and save as JSONL"""
        
        print(f"Loading data from {input_file}...")
        with open(input_file, 'r') as f:
            raw_data = json.load(f)
        
        print(f"Creating dataset with {len(raw_data)} examples...")
        dataset = Dataset.from_list(raw_data)
        
        # Tokenize
        print("Tokenizing data...")
        dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=32,
            remove_columns=dataset.column_names
        )
        
        # Split into train/val/test
        print("Splitting data...")
        test_split = 1 - train_split - val_split
        
        splits = dataset.train_test_split(test_size=val_split + test_split, seed=42)
        train_dataset = splits["train"]
        
        if test_split > 0:
            splits = splits["test"].train_test_split(test_size=test_split/(val_split + test_split), seed=42)
            val_dataset = splits["train"]
            test_dataset = splits["test"]
        else:
            val_dataset = splits["test"]
            test_dataset = None
        
        # Save datasets
        output_path = Path(output_file).parent
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_path = str(output_path / "train.jsonl")
        val_path = str(output_path / "val.jsonl")
        
        print(f"Saving datasets...")
        train_dataset.to_json(train_path)
        val_dataset.to_json(val_path)
        
        if test_dataset:
            test_path = str(output_path / "test.jsonl")
            test_dataset.to_json(test_path)
            print(f"✓ Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
        else:
            print(f"✓ Train: {len(train_dataset)} | Val: {len(val_dataset)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for fine-tuning")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b", help="Model name")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--train_split", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    
    args = parser.parse_args()
    
    processor = DataProcessor(model_name=args.model, max_seq_length=args.max_seq_length)
    processor.load_and_process(
        input_file=args.input,
        output_file=args.output,
        train_split=args.train_split,
        val_split=args.val_split
    )


if __name__ == "__main__":
    main()
