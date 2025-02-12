# train.py
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import json

token = "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"
login(token)

# Load and preprocess the dataset
def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def preprocess_data(data):
    problems = [item["problem"] for item in data]
    solutions = [item["solution"] for item in data]
    return Dataset.from_dict({"problem": problems, "solution": solutions})

# Replace with your dataset path
data = load_data("data/dataset.json")
dataset = preprocess_data(data)

# Split into train and validation sets
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

# Load the pre-trained Mistral 7B model and tokenizer
model_name = "mistralai/Mistral-7B-v0.1"  # Mistral 7B model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["problem"], examples["solution"], truncation=True, padding="max_length", max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./models",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
    learning_rate=5e-5,
    fp16=True,  # Enable mixed precision training
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./models/fine-tuned-mistral-7b")
tokenizer.save_pretrained("./models/fine-tuned-mistral-7b")