# Llama_FineTuning# Practical Implementation Guide

A comprehensive, hands-on project for fine-tuning Meta's Llama model using PyTorch and Hugging Face. Learn how to adapt a state-of-the-art LLM to your specific domain or task.

## 🎯 What You'll Learn

- **Practical Fine-Tuning:** End-to-end workflow for adapting Llama to custom tasks
- **LoRA & QLoRA:** Parameter-efficient fine-tuning techniques for limited GPU memory
- **Data Handling:** Proper dataset preprocessing, tokenization, and batching
- **Training & Evaluation:** Custom training loops, metrics, and validation strategies
- **Inference & Deployment:** Running your fine-tuned model locally and in production

## 🛠️ Tech Stack

- **Model:** Meta Llama 2 / Llama 3 (7B, 13B variants)
- **Framework:** PyTorch + Hugging Face Transformers
- **Optimization:** LoRA (Low-Rank Adaptation) & QLoRA (Quantized LoRA)
- **Training:** Accelerate, bitsandbytes (for 4-bit quantization)
- **Evaluation:** Evaluate library, custom metrics
- **Infrastructure:** Python 3.10+, CUDA 12.0+ (optional GPU support)

---

## 📥 Installation

### Prerequisites
- Python 3.10 or higher
- 16GB+ RAM (32GB+ recommended for 13B models)
- GPU (optional but highly recommended: RTX 4090, A100, or similar)
- CUDA 12.0+ (if using GPU)

### Step 1: Clone the Repository
```bash
git clone https://github.com/AdarshInturi0425/Llama-FineTuning.git
cd Llama-FineTuning
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Authenticate with Hugging Face
```bash
huggingface-cli login
# Paste your Hugging Face API token
```

---

## 🚀 Quick Start

### 1. **Prepare Your Dataset**

Place your training data in `data/raw/` in JSON format:

```json
[
  {
    "instruction": "What is the capital of France?",
    "input": "",
    "output": "Paris is the capital of France."
  },
  {
    "instruction": "Summarize this text",
    "input": "The quick brown fox...",
    "output": "A fast fox..."
  }
]
```

### 2. **Run the Data Preparation Script**
```bash
python src/prepare_data.py \
  --input data/raw/your_dataset.json \
  --output data/processed/train.jsonl \
  --split 0.8 0.1 0.1
```

### 3. **Configure Training**

Edit `configs/training_config.yaml`:
```yaml
model_name: "meta-llama/Llama-2-7b"
learning_rate: 5e-4
batch_size: 4
num_epochs: 3
use_lora: true
lora_rank: 8
use_quantization: false
```

### 4. **Start Fine-Tuning**
```bash
python src/train.py \
  --config configs/training_config.yaml \
  --data data/processed/train.jsonl \
  --output models/llama-finetuned
```

### 5. **Evaluate Your Model**
```bash
python src/evaluate.py \
  --model models/llama-finetuned \
  --data data/processed/val.jsonl
```

### 6. **Run Inference**
```bash
python src/inference.py \
  --model models/llama-finetuned \
  --prompt "What is machine learning?"
```

---

## 📂 Project Structure

```
Llama-FineTuning/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── configs/
│   ├── training_config.yaml       # Training hyperparameters
│   ├── lora_config.yaml           # LoRA configuration
│   └── quantization_config.yaml   # Quantization settings
├── src/
│   ├── prepare_data.py            # Dataset preprocessing & tokenization
│   ├── train.py                   # Main training script
│   ├── evaluate.py                # Evaluation metrics & validation
│   ├── inference.py               # Inference on fine-tuned model
│   ├── utils.py                   # Helper functions
│   └── custom_callbacks.py        # Training callbacks
├── data/
│   ├── raw/                       # Raw datasets (JSON)
│   └── processed/                 # Preprocessed datasets (JSONL)
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Exploratory data analysis
│   ├── 02_training_walkthrough.ipynb
│   └── 03_inference_examples.ipynb
├── models/                        # Fine-tuned model checkpoints
└── logs/                          # Training logs & tensorboard
```

---

## 🔧 Advanced Usage

### Fine-Tuning with LoRA (Memory Efficient)

```python
from src.train import train_with_lora

trainer = train_with_lora(
    model_name="meta-llama/Llama-2-7b",
    lora_rank=8,
    lora_alpha=32,
    dataset_path="data/processed/train.jsonl"
)
trainer.train()
```

### Fine-Tuning with QLoRA (4-bit Quantization)

```python
from src.train import train_with_qlora

trainer = train_with_qlora(
    model_name="meta-llama/Llama-2-7b",
    lora_rank=8,
    dataset_path="data/processed/train.jsonl"
)
trainer.train()
```

### Custom Evaluation Metrics

```python
from src.evaluate import CustomEvaluator

evaluator = CustomEvaluator(model_path="models/llama-finetuned")
results = evaluator.evaluate(
    data_path="data/processed/val.jsonl",
    metrics=["perplexity", "bleu", "rouge"]
)
```

---

## 📊 Training Tips & Best Practices

### Dataset Preparation
- ✅ Clean and deduplicate your data
- ✅ Ensure balanced class distribution (if classification task)
- ✅ Use 80/10/10 train/val/test split
- ✅ Aim for 1,000+ examples (minimum for good results)

### Hyperparameter Tuning
- Start with lower learning rate: `1e-4` to `5e-4`
- Use batch size of 4-8 (adjust based on GPU memory)
- Warmup steps: 10% of total training steps
- Weight decay: `0.01`

### Memory Optimization
- Use LoRA for 7B-13B models on consumer GPUs
- Use QLoRA for 70B models with 24GB+ VRAM
- Gradient checkpointing can reduce memory by 20-30%
- Use mixed precision (fp16/bf16) training

### Monitoring Training
```bash
tensorboard --logdir logs/
```
Monitor loss, learning rate, GPU memory usage in real-time.

---

## 🎓 Example Workflows

### 1. **Domain-Specific Fine-Tuning**
Fine-tune Llama on medical/legal/financial documents.

### 2. **Instruction-Following**
Adapt Llama to follow complex multi-step instructions.

### 3. **Conversational AI**
Create a domain-specific chatbot with context awareness.

### 4. **Code Generation**
Fine-tune on GitHub code for language-specific code generation.

---

## 🐛 Troubleshooting

### "CUDA out of memory"
```bash
# Solution: Use LoRA or reduce batch size
python src/train.py --use_lora true --batch_size 2
```

### "Model not found on Hugging Face Hub"
```bash
# Make sure you're logged in and have access
huggingface-cli login
huggingface-cli whoami
```

### "Slow training on CPU"
```bash
# Enable GPU acceleration
export CUDA_VISIBLE_DEVICES=0
python src/train.py --device cuda
```

---

## 📚 Learning Resources

- [Hugging Face Fine-Tuning Guide](https://huggingface.co/docs/transformers/training)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Llama Official Documentation](https://llama.meta.com/)
- [Efficient Transformers](https://huggingface.co/docs/transformers/perf_train_gpu_one)

---

## 📝 Citation

If you use this project, please cite:

```bibtex
@software{llama_finetuning_2026,
  title={Llama Fine-Tuning: Practical Implementation Guide},
  author={Inturi, Adarsh},
  year={2026},
  url={https://github.com/AdarshInturi0425/Llama-FineTuning}
}
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 💬 Questions?

Open an issue on GitHub or reach out directly. Happy fine-tuning! 🚀