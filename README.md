# Finetune-gpt2-with-google-collab
# Install required packages (if not already)
!pip install transformers datasets

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Imports
import os
import json
import torch
import gc
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load JSONL files ---
def load_jsonl_for_finetuning(file_paths):
    combined_data = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                combined_data.append(json.loads(line))
    return combined_data
    import os
os.environ["WANDB_DISABLED"] = "true"


# 🔁 Update file paths inside Google Drive
finetune_dataset_path = [
    "/content/drive/MyDrive/file_name.jsonl",
    "/content/drive/MyDrive/file_name.jsonl"
]

finetune_data = load_jsonl_for_finetuning(finetune_dataset_path)

# Save combined data as a temporary dataset for `load_dataset`
temp_combined_path = "/content/combined_data.jsonl"
with open(temp_combined_path, 'w', encoding='utf-8') as f:
    for entry in finetune_data:
        f.write(json.dumps(entry) + "\n")

dataset = load_dataset("json", data_files={"train": temp_combined_path})
train_test_split = dataset['train'].train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
validation_dataset = train_test_split['test']

print(f"Training size: {len(train_dataset)}")
print(f"Validation size: {len(validation_dataset)}")

# --- Fine-tune DialoGPT ---
model_name_dialogpt = "microsoft/DialoGPT-medium"
tokenizer_dialogpt_finetune = AutoTokenizer.from_pretrained(model_name_dialogpt)
model_dialogpt_finetune = AutoModelForCausalLM.from_pretrained(model_name_dialogpt).to(device)

# Special tokens
tokenizer_dialogpt_finetune.add_special_tokens({'pad_token': '[PAD]'})
model_dialogpt_finetune.resize_token_embeddings(len(tokenizer_dialogpt_finetune))
model_dialogpt_finetune.config.pad_token_id = tokenizer_dialogpt_finetune.convert_tokens_to_ids('[PAD]')

def tokenize_dialogpt_data(example):
    input_enc = tokenizer_dialogpt_finetune(example['input'], truncation=True, padding='max_length', max_length=50)
    output_enc = tokenizer_dialogpt_finetune(example['output'], truncation=True, padding='max_length', max_length=50)
    input_enc['labels'] = output_enc['input_ids']
    return input_enc

train_dataset_dialogpt = train_dataset.map(tokenize_dialogpt_data, batched=True)
validation_dataset_dialogpt = validation_dataset.map(tokenize_dialogpt_data, batched=True)

# 💾 Save directory in Google Drive
dialogpt_output_dir = "/content/drive/MyDrive/folder_name/fine_tuned_dialog_gpt"

training_args_dialogpt = TrainingArguments(
    output_dir=dialogpt_output_dir,
    learning_rate=5e-5,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    num_train_epochs=5,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
)

trainer_dialogpt = Trainer(
    model=model_dialogpt_finetune,
    args=training_args_dialogpt,
    train_dataset=train_dataset_dialogpt,
    eval_dataset=validation_dataset_dialogpt,
)

print("🛠️ Fine-tuning DialoGPT...")
trainer_dialogpt.train()
model_dialogpt_finetune.save_pretrained(dialogpt_output_dir)
tokenizer_dialogpt_finetune.save_pretrained(dialogpt_output_dir)
print("✅ DialoGPT fine-tuned and saved to Drive!")

# Free memory
del trainer_dialogpt
del model_dialogpt_finetune
del tokenizer_dialogpt_finetune
torch.cuda.empty_cache()
gc.collect()

# --- Fine-tune GPT-2 ---
model_name_gpt2 = "gpt2"
tokenizer_gpt2_finetune = AutoTokenizer.from_pretrained(model_name_gpt2)
model_gpt2_finetune = AutoModelForCausalLM.from_pretrained(model_name_gpt2).to(device)

# Padding fix
tokenizer_gpt2_finetune.pad_token = tokenizer_gpt2_finetune.eos_token
model_gpt2_finetune.config.pad_token_id = tokenizer_gpt2_finetune.eos_token_id

def tokenize_gpt2_data(example):
    input_enc = tokenizer_gpt2_finetune(example['input'], truncation=True, padding='max_length', max_length=100)
    output_enc = tokenizer_gpt2_finetune(example['output'], truncation=True, padding='max_length', max_length=100)
    input_enc['labels'] = output_enc['input_ids']
    return input_enc

train_dataset_gpt2 = train_dataset.map(tokenize_gpt2_data, batched=True)
validation_dataset_gpt2 = validation_dataset.map(tokenize_gpt2_data, batched=True)

gpt2_output_dir = "/content/drive/MyDrive/folder_name/fine_tuned_gpt2"

training_args_gpt2 = TrainingArguments(
    output_dir=gpt2_output_dir,
    learning_rate=5e-5,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    num_train_epochs=5,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
)

trainer_gpt2 = Trainer(
    model=model_gpt2_finetune,
    args=training_args_gpt2,
    train_dataset=train_dataset_gpt2,
    eval_dataset=validation_dataset_gpt2,
)

print("🛠️ Fine-tuning GPT-2...")
trainer_gpt2.train()
model_gpt2_finetune.save_pretrained(gpt2_output_dir)
tokenizer_gpt2_finetune.save_pretrained(gpt2_output_dir)
print("✅ GPT-2 fine-tuned and saved to Drive!")

# Clean up
del trainer_gpt2
del model_gpt2_finetune
del tokenizer_gpt2_finetune
torch.cuda.empty_cache()
gc.collect()

!ls /content/drive/MyDrive/folder_name/


# 🧠 Reload models from Drive later like this:
# tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/folder_name/fine_tuned_gpt2")
# model = AutoModelForCausalLM.from_pretrained("/content/drive/MyDrive/folder_name/fine_tuned_gpt2").to(device)
