# Raadhe-AI# Raadhe AI 🤖💛

Raadhe AI is an emotionally intelligent language model based on TinyLLaMA (1.1B), fine-tuned to be a warm, friendly, and emotionally aware conversational companion. Using LoRA (Low-Rank Adaptation), Raadhe maintains the efficiency of the base model while learning to be more empathetic and supportive.

## Features

- Emotionally intelligent responses
- Efficient fine-tuning using LoRA
- ChatML format for structured conversations
- Easy deployment options (FastAPI/Gradio)
- Modular design for future extensions

## Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/raadhe-ai.git
cd raadhe-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

To fine-tune Raadhe on your own dataset:

1. Prepare your training data in ChatML format (see `data/sample_conversations.json` for example)
2. Update training parameters in `config/train_config.yaml`
3. Run the training script:
```bash
python scripts/train.py
```

The fine-tuned LoRA adapters will be saved in `models/raadhe-lora/`.

## Testing

To test Raadhe's responses:

```bash
python scripts/generate_response.py
```

This will start an interactive session where you can chat with Raadhe.

## Project Structure

```
raadhe-ai/
├── config/           # Configuration files
├── data/            # Training and evaluation datasets
├── models/          # Saved model checkpoints
├── logs/            # Training logs
├── scripts/         # Training and inference scripts
└── deployment/      # Deployment-related code
```

## Future Enhancements

- Memory system for context-aware conversations
- RAG integration for knowledge retrieval
- Web UI for easier interaction
- Multi-turn conversation support
- Emotion detection and response adaptation

## License

MIT License - See LICENSE file for details 
