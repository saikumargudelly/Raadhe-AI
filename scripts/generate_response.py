import os
import yaml
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def load_config():
    with open("config/train_config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_model_and_tokenizer(config):
    # Load base model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["base_model"],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["base_model"],
        padding_side="right",
        use_fast=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        config["output"]["output_dir"],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    return model, tokenizer

def format_prompt(messages):
    formatted_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted_text += f"<|{role}|>\n{content}\n"
    return formatted_text

def generate_response(model, tokenizer, messages, max_length=2048):
    # Format the conversation
    prompt = format_prompt(messages)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return response
   # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
   # return response

def main():
    # Load configuration
    config = load_config()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    print("Raadhe AI is ready to chat! Type 'quit' to exit.")
    print("----------------------------------------")
    
    # Initialize conversation
    messages = [
        {
            "role": "system",
            "content": "You are Raadhe, a warm, kind, and emotionally intelligent AI friend. You're empathetic, supportive, and always ready to listen. You use emojis naturally and speak in a friendly, conversational way."
        }
    ]
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == "quit":
            print("\nGoodbye! Take care! 💛")
            break
        
        # Add user message
        messages.append({"role": "user", "content": user_input})
        
        # Generate response
        response = generate_response(model, tokenizer, messages)
        
        # Extract assistant's response
        assistant_response = response.split("<|assistant|>")[-1].strip()
        
        # Add assistant response to messages
        messages.append({"role": "assistant", "content": assistant_response})
        
        # Print response
        print(f"\nRaadhe: {assistant_response}")

if __name__ == "__main__":
    main() 