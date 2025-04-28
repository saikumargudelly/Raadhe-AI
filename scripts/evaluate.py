import os
import yaml
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def evaluate_model(model, tokenizer, eval_dataset):
    results = {
        "responses": [],
        "references": [],
        "metrics": {}
    }
    
    for example in tqdm(eval_dataset, desc="Evaluating"):
        # Get conversation history
        messages = example["messages"][:-1]  # Exclude the last message (reference)
        reference = example["messages"][-1]["content"]
        
        # Generate response
        response = generate_response(model, tokenizer, messages)
        assistant_response = response.split("<|assistant|>")[-1].strip()
        
        # Store results
        results["responses"].append(assistant_response)
        results["references"].append(reference)
    
    # Calculate metrics
    # Note: This is a simple implementation. You might want to use more sophisticated metrics
    # like BLEU, ROUGE, or semantic similarity scores
    results["metrics"]["response_length"] = {
        "mean": np.mean([len(r) for r in results["responses"]]),
        "std": np.std([len(r) for r in results["responses"]])
    }
    
    return results

def save_results(results, output_file):
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

def main():
    # Load configuration
    config = load_config()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Load evaluation dataset
    eval_dataset = load_dataset("json", data_files=config["data"]["eval_file"])["train"]
    
    # Evaluate model
    results = evaluate_model(model, tokenizer, eval_dataset)
    
    # Save results
    output_file = os.path.join(config["output"]["output_dir"], "evaluation_results.json")
    save_results(results, output_file)
    
    # Print summary
    print("\nEvaluation Results:")
    print("------------------")
    print(f"Number of examples evaluated: {len(results['responses'])}")
    print(f"Average response length: {results['metrics']['response_length']['mean']:.2f} characters")
    print(f"Response length std dev: {results['metrics']['response_length']['std']:.2f} characters")
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main() 