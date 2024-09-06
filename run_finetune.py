
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Load the LLaMA 3.1-8B Instruct model from Huggingface
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load your custom dataset (JSON format)
dataset = load_dataset("json", data_files="./data/combined_dataset_for_finetuning.json")

# Preprocessing the dataset - tokenizing input texts
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=1,  # Adjust based on GPU memory
    per_device_eval_batch_size=1,   # Adjust for validation
    num_train_epochs=3,             # Modify as necessary
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,                      # Mixed precision for faster training
)

# Initialize the Trainer class
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Start fine-tuning
trainer.train()
