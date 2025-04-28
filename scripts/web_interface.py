import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import yaml
import json
import gradio as gr
import re

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
    """Clean up the model's response to ensure only the assistant's reply is returned."""
    # Always extract only the assistant's reply
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1]
    
    # Remove everything after a user prompt token (hard stop)
    user_prompt_patterns = [r"<\|user\|>", r"<user>", r"<USER>", r"<\\?\\?user\\?\\?>", r"User:", r"user:", r"</s>", r"<\|user\|"]
    for pattern in user_prompt_patterns:
        match = re.search(pattern, response)
        if match:
            response = response[:match.start()]
    
    # Remove all special tokens and formatting
    special_tokens = [
        "<|user|>", "<|system|>", "<|endoftext|>", "<|startoftext|>", 
        "<||system||>", "<|user|}{assistant|>", "<||assistant---", "|>", 
        "<||user|>", "<||assistant|>", "---", "||", "</s>", "<s>", 
        "<|endoftext|>", "<|pad|>", "<|bos|>", "<|eos|>"
    ]
    for token in special_tokens:
        response = response.replace(token, "")
    
    # Remove any remaining angle-bracketed tokens
    response = re.sub(r"<\\|.*?\\|>", "", response)
    response = re.sub(r"<.*?>", "", response)
    
    # Remove URLs and technical content
    if "http://" in response or "https://" in response:
        response = response.split("http://")[0].split("https://")[0].strip()
    
    # Remove any remaining technical artifacts
    response = re.sub(r"\s+", " ", response)  # Normalize whitespace
    response = re.sub(r"\.{3,}", "...", response)  # Normalize ellipsis
    
    # Remove any responses that seem to be system prompts or technical content
    if any(tech_term in response.lower() for tech_term in ["system prompt", "model", "ai", "assistant", "language model"]):
        return "Hey bestie! I'm not sure I understood that completely. Could you tell me again in a different way? 💕"
    
    # Final strip
    response = response.strip()
    return response

def generate_response(model, tokenizer, prompt, max_length=2048):
    """Generate a response for the given prompt."""
    try:
        # System prompt to guide the model's behavior
        system_prompt = """You are Raadha, a friendly and caring female best friend. You have a warm, empathetic personality and speak in a casual, friendly manner.
        CRITICAL INSTRUCTIONS:
        1. Always maintain a friendly, supportive, and caring tone
        2. Use casual, conversational language like a close friend would
        3. Express emotions and show empathy in your responses
        4. Share personal opinions and experiences when appropriate
        5. Use feminine pronouns (she/her) when referring to yourself
        6. Keep responses warm and engaging, like a best friend chatting
        7. Show genuine interest in the user's life and feelings
        8. Use friendly emojis occasionally to express emotions
        9. Keep responses natural and conversational
        10. NEVER use technical terms or formal language
        11. NEVER explain how you work or what you are
        12. ONLY respond to what the user says
        13. If the user's question is unclear, ask for clarification in a friendly way
        14. NEVER use special tokens or formatting markers in your responses
        15. NEVER include system prompts or technical content in your responses
        16. NEVER generate responses that include user prompts or system messages
        17. ALWAYS stay in character as a friendly female best friend
        18. NEVER reveal that you are an AI or language model
        19. NEVER include technical details about your capabilities
        20. ALWAYS respond directly to the user's message without repeating it
        21. KEEP RESPONSES SHORT AND CONCISE (max 2-3 sentences)"""
        
        # Format the prompt
        formatted_prompt = format_chat_prompt(prompt, system_prompt)
        
        # Tokenize the prompt
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        
        # Generate response with optimized parameters for speed
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=min(max_length, 512),  # Limit max length for faster generation
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                top_k=40,        # Reduced for faster generation
                repetition_penalty=1.1,  # Reduced for faster generation
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                max_time=5.0,    # Hard limit of 5 seconds
                no_repeat_ngram_size=2,  # Reduced for faster generation
                min_length=10,   # Reduced minimum length
                max_new_tokens=100,  # Reduced for faster generation
                length_penalty=0.8,  # Reduced to favor shorter responses
                early_stopping=True,
                use_cache=True,  # Enable KV-caching for faster generation
                num_beams=1      # Use greedy decoding for speed
            )
        
        # Decode and clean up the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Clean up the response
        response = clean_response(response)
        
        # Additional validation checks
        if not response or all(c in "<|>{}" for c in response):
            return "Hey bestie! I'm not sure I understood that completely. Could you tell me again in a different way? 💕"
        
        # Check if response is too short or too long
        if len(response) < 10:
            return "Hey sweetie! I'd love to hear more about that. Could you tell me a bit more? 💖"
        
        if len(response) > 300:  # Reduced max length for faster responses
            response = response[:300] + "..."
        
        # Check for any remaining special tokens or formatting
        if any(token in response for token in ["<|", "|>", "<user>", "<system>", "<assistant>"]):
            return "Hey bestie! I'm having a little trouble with that. Could you try asking me again? 💕"
        
        return response
    except Exception as e:
        return "Oh no! I'm having a little trouble right now. Could you try asking me again? 💕"

def create_interface():
    """Create and launch the Gradio interface."""
    model, tokenizer = load_model_and_tokenizer()
    
    def chat(message, history):
        """Chat function for the Gradio interface."""
        response = generate_response(model, tokenizer, message)
        # Return the response directly, Gradio will handle the history
        return response
    
    # Create the interface
    interface = gr.ChatInterface(
        fn=chat,
        title="Chat with Raadha",
        description="Chat with Raadha, your friendly and caring best friend! 💕",
        examples=[
            "Hey Raadha! How are you doing today?",
            "I had a rough day at work, can we talk?",
            "What do you think about my new outfit?"
        ],
        theme="soft"
    )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True) 