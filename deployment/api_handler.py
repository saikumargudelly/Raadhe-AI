from typing import List, Dict, Any, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from fastapi import HTTPException
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RaadheAPIHandler:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        max_length: int = 2048,
        default_temperature: float = 0.7,
        default_top_p: float = 0.9
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.request_count = 0
        self.start_time = datetime.now()
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate a response from the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_length: Maximum length of the generated response
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Dictionary containing the response and metadata
        """
        try:
            start_time = time.time()
            self.request_count += 1
            
            # Use default values if not provided
            max_length = min(max_length or self.max_length, 512)  # Limit max length for faster generation
            temperature = temperature or self.default_temperature
            top_p = top_p or self.default_top_p
            
            # Format messages
            formatted_text = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                formatted_text += f"<|{role}|>\n{content}\n"
            formatted_text += "<|endoftext|>"
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                max_length=max_length,
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Generate response with optimized parameters for speed
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Reduced for faster generation
                    no_repeat_ngram_size=2,  # Reduced for faster generation
                    length_penalty=0.8,  # Reduced to favor shorter responses
                    early_stopping=True,
                    max_time=5.0,  # Hard limit of 5 seconds
                    min_length=10,  # Reduced minimum length
                    max_new_tokens=100,  # Reduced for faster generation
                    use_cache=True,  # Enable KV-caching for faster generation
                    num_beams=1  # Use greedy decoding for speed
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean and validate response
            assistant_response = response.split("<|assistant|>")[-1].strip()
            
            # Remove any remaining special tokens
            special_tokens = ["<|user|>", "<|system|>", "<|endoftext|>", "<|startoftext|>"]
            for token in special_tokens:
                assistant_response = assistant_response.replace(token, "")
            
            # Remove any responses that seem to be system prompts or technical content
            if any(tech_term in assistant_response.lower() for tech_term in ["system prompt", "model", "ai", "assistant", "language model"]):
                assistant_response = "Hey bestie! I'm not sure I understood that completely. Could you tell me again in a different way? 💕"
            
            # Validate response length
            if len(assistant_response) < 10:
                assistant_response = "Hey sweetie! I'd love to hear more about that. Could you tell me a bit more? 💖"
            elif len(assistant_response) > 300:  # Reduced max length for faster responses
                assistant_response = assistant_response[:300] + "..."
            
            # Check for any remaining special tokens or formatting
            if any(token in assistant_response for token in ["<|", "|>", "<user>", "<system>", "<assistant>"]):
                assistant_response = "Hey bestie! I'm having a little trouble with that. Could you try asking me again? 💕"
            
            # Calculate metrics
            generation_time = time.time() - start_time
            response_length = len(assistant_response)
            
            # Log request
            logger.info(
                f"Request #{self.request_count} - "
                f"Time: {generation_time:.2f}s - "
                f"Length: {response_length} chars"
            )
            
            return {
                "response": assistant_response,
                "metadata": {
                    "generation_time": generation_time,
                    "response_length": response_length,
                    "request_count": self.request_count,
                    "uptime": str(datetime.now() - self.start_time)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        return {
            "total_requests": self.request_count,
            "uptime": str(datetime.now() - self.start_time),
            "model_info": {
                "max_length": self.max_length,
                "default_temperature": self.default_temperature,
                "default_top_p": self.default_top_p
            }
        }
    
    def reset_stats(self) -> None:
        """Reset API usage statistics."""
        self.request_count = 0
        self.start_time = datetime.now()
        logger.info("API statistics reset") 