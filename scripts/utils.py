import os
import json
import yaml
import torch
from typing import List, Dict, Any, Optional
from transformers import PreTrainedTokenizer

def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def load_json(file_path: str) -> Dict[str, Any]:
    """Load a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save data to a JSON file."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

def format_chatml(messages: List[Dict[str, str]]) -> str:
    """Format a list of messages in ChatML format."""
    formatted_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted_text += f"<|{role}|>\n{content}\n"
    formatted_text += "<|endoftext|>"
    return formatted_text

def parse_chatml(text: str) -> List[Dict[str, str]]:
    """Parse ChatML formatted text into a list of messages."""
    messages = []
    current_role = None
    current_content = []
    
    for line in text.split("\n"):
        if line.startswith("<|") and line.endswith("|>"):
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": "\n".join(current_content).strip()
                })
            current_role = line[2:-2]
            current_content = []
        else:
            current_content.append(line)
    
    if current_role and current_content:
        messages.append({
            "role": current_role,
            "content": "\n".join(current_content).strip()
        })
    
    return messages

def prepare_input_for_model(
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    max_length: Optional[int] = None,
    padding: bool = True,
    truncation: bool = True
) -> Dict[str, torch.Tensor]:
    """Prepare input for the model by tokenizing and formatting messages."""
    formatted_text = format_chatml(messages)
    return tokenizer(
        formatted_text,
        return_tensors="pt",
        max_length=max_length,
        padding=padding,
        truncation=truncation
    )

def extract_assistant_response(text: str) -> str:
    """Extract the assistant's response from the generated text."""
    try:
        return text.split("<|assistant|>")[-1].strip()
    except IndexError:
        return text.strip()

def create_directory_if_not_exists(directory: str) -> None:
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_device() -> torch.device:
    """Get the appropriate device for model training/inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count the number of trainable and non-trainable parameters in a model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
        "total": trainable_params + non_trainable_params
    } 