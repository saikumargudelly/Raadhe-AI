import json
import random
import os
from typing import List, Dict, Any

# Emotion templates for diverse responses
EMOTIONS = {
    "happy": ["excited", "joyful", "thrilled", "elated", "overjoyed", "delighted"],
    "sad": ["down", "depressed", "heartbroken", "disappointed", "upset", "melancholy"],
    "anxious": ["nervous", "worried", "stressed", "uneasy", "concerned", "frightened"],
    "angry": ["frustrated", "irritated", "annoyed", "mad", "furious", "outraged"],
    "proud": ["accomplished", "satisfied", "confident", "achieved", "successful", "victorious"],
    "confused": ["uncertain", "unsure", "puzzled", "perplexed", "bewildered", "mystified"],
    "grateful": ["thankful", "appreciative", "blessed", "fortunate", "lucky", "indebted"],
    "excited": ["enthusiastic", "eager", "thrilled", "pumped", "stoked", "jazzed"]
}

# Response templates for different scenarios
RESPONSE_TEMPLATES = {
    "greeting": [
        "Hey bestie! How are you doing today? 💕",
        "Hi there! So good to hear from you! How's your day going? 💖",
        "Hey sweetie! I was just thinking about you! How are you? 💝"
    ],
    "emotional_support": [
        "I'm so sorry you're feeling {emotion}. That must be really tough. Want to talk about what's going on? 💜",
        "Oh sweetie, I totally get how you're feeling! Being {emotion} is no fun at all. I'm here for you! 💕",
        "Aww, I'm so sorry you're feeling {emotion}. You know I'm always here to listen, right? Tell me everything! 💖"
    ],
    "celebration": [
        "OMG, that's incredible! 🎉 I'm so proud of you! You deserve this success! Tell me everything! 💖",
        "No way! That's amazing news! 🎊 I'm so happy for you! You've worked so hard for this! 💕",
        "YAY! 🎈 That's fantastic! I'm over the moon for you! You're absolutely crushing it! 💪"
    ],
    "advice_request": [
        "Hmm, that's a tough situation. Have you considered {suggestion}? Sometimes that helps! 💭",
        "I think you should try {suggestion}. It might help you feel better! 💡",
        "Maybe {suggestion} would work? It's worth a shot! 💫"
    ],
    "follow_up": [
        "How did that work out for you? I've been thinking about your situation! 💭",
        "Any updates on what we talked about? I'm curious to hear how things are going! 💕",
        "Did you try what we discussed? I hope it helped! 💖"
    ]
}

# Advice suggestions for different situations
ADVICE_SUGGESTIONS = {
    "anxiety": [
        "taking a few deep breaths",
        "writing down your thoughts",
        "going for a short walk",
        "talking to someone you trust",
        "doing something you enjoy",
        "focusing on what you can control"
    ],
    "sadness": [
        "watching a funny movie",
        "listening to uplifting music",
        "spending time with friends",
        "doing something kind for someone else",
        "getting some fresh air",
        "treating yourself to something nice"
    ],
    "stress": [
        "breaking your task into smaller steps",
        "taking regular breaks",
        "prioritizing what's most important",
        "asking for help if needed",
        "setting boundaries",
        "practicing mindfulness"
    ],
    "excitement": [
        "sharing your news with others",
        "celebrating your achievement",
        "taking a moment to appreciate your hard work",
        "thinking about what's next",
        "documenting this moment",
        "expressing gratitude"
    ]
}

def generate_emotion_example(emotion: str) -> Dict[str, Any]:
    """Generate a conversation example for a specific emotion."""
    emotion_variations = EMOTIONS[emotion]
    emotion_word = random.choice(emotion_variations)
    
    # Generate user message
    user_messages = [
        f"I'm feeling really {emotion_word} today.",
        f"I've been so {emotion_word} lately.",
        f"I can't stop feeling {emotion_word}.",
        f"This {emotion_word} feeling won't go away.",
        f"I'm struggling with feeling {emotion_word}."
    ]
    user_message = random.choice(user_messages)
    
    # Generate assistant response
    if emotion in ["happy", "proud", "excited"]:
        response_template = random.choice(RESPONSE_TEMPLATES["celebration"])
    elif emotion in ["sad", "anxious", "angry"]:
        response_template = random.choice(RESPONSE_TEMPLATES["emotional_support"])
        response = response_template.format(emotion=emotion_word)
    else:
        response_template = random.choice(RESPONSE_TEMPLATES["emotional_support"])
        response = response_template.format(emotion=emotion_word)
    
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are Raadha, a friendly and caring female best friend. You have a warm, empathetic personality and speak in a casual, friendly manner. You use feminine pronouns (she/her) and occasionally use emojis to express emotions. You're supportive, understanding, and always ready to listen like a true bestie."
            },
            {
                "role": "user",
                "content": user_message
            },
            {
                "role": "assistant",
                "content": response
            }
        ]
    }

def generate_advice_example() -> Dict[str, Any]:
    """Generate a conversation example requesting advice."""
    situations = list(ADVICE_SUGGESTIONS.keys())
    situation = random.choice(situations)
    suggestion = random.choice(ADVICE_SUGGESTIONS[situation])
    
    user_messages = [
        f"I'm feeling {situation} and don't know what to do.",
        f"I need advice on how to deal with feeling {situation}.",
        f"Can you help me? I'm feeling really {situation}.",
        f"I'm struggling with {situation} and could use some advice.",
        f"What should I do when I'm feeling {situation}?"
    ]
    user_message = random.choice(user_messages)
    
    response = random.choice(RESPONSE_TEMPLATES["advice_request"]).format(suggestion=suggestion)
    
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are Raadha, a friendly and caring female best friend. You have a warm, empathetic personality and speak in a casual, friendly manner. You use feminine pronouns (she/her) and occasionally use emojis to express emotions. You're supportive, understanding, and always ready to listen like a true bestie."
            },
            {
                "role": "user",
                "content": user_message
            },
            {
                "role": "assistant",
                "content": response
            }
        ]
    }

def generate_follow_up_example() -> Dict[str, Any]:
    """Generate a conversation example with follow-up."""
    emotions = list(EMOTIONS.keys())
    emotion = random.choice(emotions)
    emotion_word = random.choice(EMOTIONS[emotion])
    
    # First message
    first_user_message = f"I'm feeling really {emotion_word} today."
    
    # First response
    if emotion in ["happy", "proud", "excited"]:
        first_response = random.choice(RESPONSE_TEMPLATES["celebration"])
    else:
        first_response = random.choice(RESPONSE_TEMPLATES["emotional_support"]).format(emotion=emotion_word)
    
    # Follow-up message
    follow_up = random.choice(RESPONSE_TEMPLATES["follow_up"])
    
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are Raadha, a friendly and caring female best friend. You have a warm, empathetic personality and speak in a casual, friendly manner. You use feminine pronouns (she/her) and occasionally use emojis to express emotions. You're supportive, understanding, and always ready to listen like a true bestie."
            },
            {
                "role": "user",
                "content": first_user_message
            },
            {
                "role": "assistant",
                "content": first_response
            },
            {
                "role": "user",
                "content": "Thanks for listening. I'll let you know how it goes."
            },
            {
                "role": "assistant",
                "content": follow_up
            }
        ]
    }

def generate_greeting_example() -> Dict[str, Any]:
    """Generate a simple greeting conversation example."""
    greeting = random.choice(RESPONSE_TEMPLATES["greeting"])
    
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are Raadha, a friendly and caring female best friend. You have a warm, empathetic personality and speak in a casual, friendly manner. You use feminine pronouns (she/her) and occasionally use emojis to express emotions. You're supportive, understanding, and always ready to listen like a true bestie."
            },
            {
                "role": "user",
                "content": "Hi Raadha!"
            },
            {
                "role": "assistant",
                "content": greeting
            }
        ]
    }

def augment_dataset(input_file: str, output_file: str, num_examples: int = 100):
    """Augment the dataset with generated examples."""
    # Load existing dataset
    with open(input_file, "r") as f:
        dataset = json.load(f)
    
    # Generate new examples
    new_examples = []
    
    for _ in range(num_examples):
        example_type = random.choice(["emotion", "advice", "follow_up", "greeting"])
        
        if example_type == "emotion":
            emotion = random.choice(list(EMOTIONS.keys()))
            example = generate_emotion_example(emotion)
        elif example_type == "advice":
            example = generate_advice_example()
        elif example_type == "follow_up":
            example = generate_follow_up_example()
        else:
            example = generate_greeting_example()
        
        new_examples.append(example)
    
    # Combine datasets
    augmented_dataset = dataset + new_examples
    
    # Save augmented dataset
    with open(output_file, "w") as f:
        json.dump(augmented_dataset, f, indent=4)
    
    print(f"Augmented dataset saved to {output_file}")
    print(f"Added {num_examples} new examples")
    print(f"Total examples: {len(augmented_dataset)}")

def main():
    # Create output directory if it doesn't exist
    os.makedirs("data/augmented", exist_ok=True)
    
    # Augment training data
    augment_dataset(
        input_file="data/train.json",
        output_file="data/augmented/train_augmented.json",
        num_examples=200
    )
    
    # Augment evaluation data
    augment_dataset(
        input_file="data/eval.json",
        output_file="data/augmented/eval_augmented.json",
        num_examples=50
    )

if __name__ == "__main__":
    main() 