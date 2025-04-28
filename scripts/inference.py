import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import yaml
import json
import sys

def load_config(config_path: str = "config/train_config.yaml") -> dict:
    """Load training configuration."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_lora_config(config_path: str = "config/lora_config.json") -> dict:
    """Load LoRA configuration."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def load_model_and_tokenizer():
    """Load the base model, tokenizer, and LoRA weights."""
    config = load_config()
    lora_config = load_lora_config()
    
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model"],
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["base_model"],
        padding_side="right",
        use_fast=True
    )
    
    # Add special tokens
    special_tokens = {
        "pad_token": "<|pad|>",
        "eos_token": "<|endoftext|>",
        "bos_token": "<|startoftext|>"
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    try:
        model = PeftModel.from_pretrained(model, config["output"]["output_dir"])
    except Exception as e:
        print(f"Error loading LoRA weights: {e}")
    
    return model, tokenizer

def format_chat_prompt(prompt: str, system_prompt: str = None) -> str:
    """Format the chat prompt with system message if provided."""
    if system_prompt:
        return f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
    return f"<|user|>\n{prompt}\n<|assistant|>\n"

def clean_response(response: str) -> str:
    """Clean up the model's response."""
    # Remove any system or user messages that might be in the response
    if "<|system|>" in response:
        response = response.split("<|system|>")[-1]
    if "<|user|>" in response:
        response = response.split("<|user|>")[-1]
    
    # Extract assistant's response
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1]
    
    # Clean up the response
    response = response.strip()
    response = response.replace("<|endoftext|>", "").strip()
    
    # Remove any remaining special tokens
    response = response.replace("<|user|>", "").replace("<|assistant|>", "")
    response = response.replace("<||system||>", "").replace("<|user|}{assistant}|", "")
    response = response.replace("<||assistant---", "").replace("|>", "")
    
    # Remove URLs and technical content
    if "http://" in response or "https://" in response:
        response = response.split("http://")[0].strip()
    
    return response

def generate_response(model, tokenizer, prompt, max_length=2048):
    """Generate a response for the given prompt."""
    try:
        # System prompt to guide the model's behavior
        system_prompt = """You are Raadhe, a helpful and friendly AI assistant. 
        You should provide clear, concise, and relevant responses. 
        Always be polite and professional."""
        
        # Format the prompt
        formatted_prompt = format_chat_prompt(prompt, system_prompt)
        
        # Tokenize the prompt
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                max_time=30.0,
                no_repeat_ngram_size=3
            )
        
        # Decode and clean up the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = clean_response(response)
        
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    """Main function to run the model."""
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer()
        print("\nModel loaded successfully! You can now start chatting.")
        print("Type 'quit' to exit.\n")
        
        # Interactive chat loop
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break
                
            response = generate_response(model, tokenizer, user_input)
            print(f"Assistant: {response}\n")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 