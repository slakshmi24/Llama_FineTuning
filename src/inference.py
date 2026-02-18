"""
Inference Script for Fine-Tuned Llama Models
Generate text using your fine-tuned model
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class LlamaInference:
    """Run inference on fine-tuned Llama models"""
    
    def __init__(self, model_path: str, use_lora: bool = True, device: str = "cuda"):
        self.device = device
        self.model_path = model_path
        self.use_lora = use_lora
        
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load fine-tuned model"""
        print(f"Loading model from {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        )
        
        # Load LoRA weights if applicable
        if self.use_lora:
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            self.model = base_model
        
        self.model.eval()
    
    def generate(self, prompt: str, max_length: int = 512, temperature: float = 0.7,
                top_p: float = 0.9) -> str:
        """Generate text given a prompt"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def chat(self, instruction: str, input_text: str = "") -> str:
        """Chat-style interaction"""
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        return self.generate(prompt)


def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Llama")
    parser.add_argument("--model", required=True, help="Path to fine-tuned model")
    parser.add_argument("--prompt", help="Single prompt for generation")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--max_length", type=int, default=512, help="Max generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Load model
    inference = LlamaInference(args.model, device=args.device)
    
    if args.prompt:
        # Single generation
        response = inference.generate(args.prompt, max_length=args.max_length,
                                     temperature=args.temperature)
        print(f"\n{response}\n")
    
    elif args.interactive:
        # Interactive chat
        print("Starting interactive mode (type 'quit' to exit)...")
        while True:
            instruction = input("\n📝 Instruction: ").strip()
            if instruction.lower() == "quit":
                break
            
            input_text = input("📎 Input (optional): ").strip()
            
            response = inference.chat(instruction, input_text)
            print(f"\n✨ Response:\n{response}")
    
    else:
        # Interactive prompt if no prompt provided
        print("Usage: python inference.py --model <path> [--prompt <text>] [--interactive]")


if __name__ == "__main__":
    main()