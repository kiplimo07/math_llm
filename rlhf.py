# rlhf.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig

# Load the fine-tuned model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./models/fine-tuned-llama-3.2")
tokenizer = AutoTokenizer.from_pretrained("./models/fine-tuned-llama-3.2")

# Set up PPO configuration
config = PPOConfig(
    model_name="./models/fine-tuned-llama-3.2",
    learning_rate=1e-5,
    batch_size=32,
)

# Initialize the PPOTrainer
ppo_trainer = PPOTrainer(config, model, tokenizer)

# Train with RLHF
ppo_trainer.train()