"""
Fine-Tuning Script for Llama Models
Supports LoRA and QLoRA training
"""

import argparse
import yaml
from pathlib import Path
from typing import Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import wandb


class LlamaFineTuner:
    """Fine-tune Llama models with LoRA/QLoRA"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["model_name"]
        self.output_dir = config.get("output_dir", "models/llama-finetuned")
        self.use_lora = config.get("use_lora", True)
        self.use_quantization = config.get("use_quantization", False)
        
        self.tokenizer = None
        self.model = None
    
    def load_model(self):
        """Load model with optional quantization"""
        print(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization config for QLoRA
        bnb_config = None
        if self.use_quantization:
            print("Using 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )
        
        # Apply LoRA
        if self.use_lora:
            print("Applying LoRA...")
            self.model = self._apply_lora()
        
        return self.model
    
    def _apply_lora(self):
        """Apply LoRA to the model"""
        lora_config = LoraConfig(
            r=self.config.get("lora_rank", 8),
            lora_alpha=self.config.get("lora_alpha", 32),
            lora_dropout=self.config.get("lora_dropout", 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"]
        )
        
        return get_peft_model(self.model, lora_config)
    
    def train(self, train_data_path: str, val_data_path: Optional[str] = None):
        """Fine-tune the model"""
        
        # Load datasets
        print(f"Loading training data from {train_data_path}")
        train_dataset = load_dataset("json", data_files=train_data_path, split="train")
        
        eval_dataset = None
        if val_data_path:
            print(f"Loading validation data from {val_data_path}")
            eval_dataset = load_dataset("json", data_files=val_data_path, split="train")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.config.get("num_epochs", 3),
            per_device_train_batch_size=self.config.get("batch_size", 4),
            per_device_eval_batch_size=self.config.get("batch_size", 4),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 1),
            learning_rate=self.config.get("learning_rate", 5e-4),
            warmup_steps=self.config.get("warmup_steps", 100),
            weight_decay=self.config.get("weight_decay", 0.01),
            logging_steps=10,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.get("eval_steps", 100),
            save_strategy="steps",
            save_steps=self.config.get("save_steps", 100),
            load_best_model_at_end=True if eval_dataset else False,
            report_to=["wandb"] if self.config.get("use_wandb", False) else [],
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            gradient_checkpointing=self.config.get("gradient_checkpointing", True)
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save final model
        print(f"Saving model to {self.output_dir}")
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        return trainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Llama models")
    parser.add_argument("--config", required=True, help="Training config YAML file")
    parser.add_argument("--train_data", required=True, help="Training data JSONL file")
    parser.add_argument("--val_data", help="Validation data JSONL file")
    parser.add_argument("--output", help="Output directory (overrides config)")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.output:
        config["output_dir"] = args.output
    
    # Fine-tune
    finetuner = LlamaFineTuner(config)
    finetuner.load_model()
    finetuner.train(args.train_data, args.val_data)


if __name__ == "__main__":
    main()