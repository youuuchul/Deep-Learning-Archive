import os
import json
import logging
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback
)
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
import evaluate
import sys
import unicodedata

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device Detection
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
logger.info(f"Using device: {device}")

is_mac_local = (device.type == 'mps')
is_colab = ('google.colab' in sys.modules)

if is_mac_local:
    logger.warning("Running on Mac M1 (MPS). Training will be limited to debugging samples.")

# Data Loader Logic
label_mapping = {"-1": 0, "0": 1, "1": 2}

def load_and_split_data(data_dir, sample_limit=None):
    files = list(Path(data_dir).glob("**/*.json"))
    data = []
    
    logger.info(f"Found {len(files)} files.")
    
    for fpath in files:
        try:
            with open(fpath, encoding='utf-8') as f:
                content = json.load(f)
                items = content if isinstance(content, list) else []
                for item in items:
                    if item.get("RawText") and item.get("GeneralPolarity") in label_mapping:
                        data.append({
                            "text": item.get("RawText"),
                            "label": label_mapping[item.get("GeneralPolarity")],
                            "product_name": item.get("ProductName")
                        })
        except Exception as e:
            continue
            
    if sample_limit:
        data = data[:sample_limit]
        logger.info(f"Sampling {len(data)} items for debugging.")

    df = pd.DataFrame(data)
    
    # Leakage Proof Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['product_name']))
    
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    
    # Overlap Check
    overlap = set(train_df.product_name) & set(test_df.product_name)
    if overlap:
        logger.warning(f"Leakage detected! {len(overlap)} products overlap.")
    else:
        logger.info("No Leakage detected.")
        
    return DatasetDict({
        "train": Dataset.from_pandas(train_df[['text', 'label']], preserve_index=False),
        "test": Dataset.from_pandas(test_df[['text', 'label']], preserve_index=False)
    })

def find_data_dir():
    if is_colab:
        return Path("/content/data")
        
    # Candidates for data root
    candidates = [Path("./data"), Path("../data"), Path("../../data")]
    
    for cand in candidates:
        if cand.exists():
            logger.info(f"Checking candidate: {cand.resolve()}")
            # Try to find specific fashion folder handling NFD/NFC
            try:
                # Debugging Directory Walk
                for p in cand.glob("*"):
                    logger.info(f"  - {p.name} (NFC: {unicodedata.normalize('NFC', p.name) == p.name})")

                shopping_dir = next(cand.glob("*쇼핑몰*"))
                logger.info(f"Found Shopping Dir: {shopping_dir}")
                
                # Debugging Shopping Dir
                logger.info("Inspecting Shopping Dir Contents:")
                for p in shopping_dir.glob("*"):
                    logger.info(f"  - {p.name} (Len: {len(p.name)}, Raw: {[ord(c) for c in p.name]})")
                    nl_name = unicodedata.normalize('NFC', p.name)
                    if "패션" in nl_name:
                        logger.info(f"    -> Normalized match found! {p}")
                
                # Robust find
                fashion_dir = None
                for p in shopping_dir.glob("*"):
                    if "패션" in unicodedata.normalize('NFC', p.name):
                        fashion_dir = p
                        break
                
                if fashion_dir:
                    logger.info(f"Found Fashion Dir: {fashion_dir}")
                    return fashion_dir
                else:
                     raise StopIteration
            except StopIteration:
                logger.warning(f"Could not find specific Fashion folder in {cand}")
                continue
                
    return None

DATA_DIR = find_data_dir()

if DATA_DIR is None:
    raise FileNotFoundError("Could not locate 'data/쇼핑몰/01. 패션' directory. Please check file structure.")
    
logger.info(f"Target Data Directory: {DATA_DIR}")

SAMPLE_LIMIT = 2000 if is_mac_local else None # Full training on Colab

# Load
logger.info(f"Loading data from {DATA_DIR}...")
dataset = load_and_split_data(DATA_DIR, sample_limit=SAMPLE_LIMIT)
print(dataset)

if len(dataset['train']) == 0:
    raise ValueError("Dataset is empty! Check data path and JSON structure.")

# 3. Preprocessing
logger.info("Starting Preprocessing...")
MODEL_ID = "klue/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding=False)

try:
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    logger.info("Preprocessing complete.")
except Exception as e:
    logger.error(f"Preprocessing Failed: {e}")
    raise e

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    f1_score = f1.compute(predictions=predictions, references=labels, average="macro")
    return {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}

# 4. Full Fine-Tuning Setup
logger.info("Initializing Full Fine-Tuning Trainer...")
model_ft = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=3)

training_args = TrainingArguments(
    output_dir="./results_debug/full_ft",
    learning_rate=2e-5,
    per_device_train_batch_size=2, # Small batch for debug
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    max_steps=2, # Only 2 steps for debugging
    weight_decay=0.01,
    eval_strategy="no",
    save_strategy="no",
    use_mps_device=is_mac_local,
    report_to="none"
)

trainer_ft = Trainer(
    model=model_ft,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

logger.info("Running Debug Training (Full FT)...")
try:
    trainer_ft.train()
    logger.info("Full FT Debug Run Success!")
except Exception as e:
    logger.error(f"Full FT Training Failed: {e}")
    raise e

# 5. PEFT (LoRA) Setup
logger.info("Initializing PEFT (LoRA) Trainer...")
model_lora = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=3)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"]
)

model_lora = get_peft_model(model_lora, lora_config)
model_lora.print_trainable_parameters()

training_args_lora = TrainingArguments(
    output_dir="./results_debug/lora",
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    max_steps=2, # Only 2 steps
    weight_decay=0.01,
    eval_strategy="no",
    save_strategy="no",
    use_mps_device=is_mac_local,
    report_to="none"
)

trainer_lora = Trainer(
    model=model_lora,
    args=training_args_lora,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

logger.info("Running Debug Training (LoRA)...")
try:
    trainer_lora.train()
    logger.info("LoRA Debug Run Success!")
except Exception as e:
    logger.error(f"LoRA Training Failed: {e}")
    raise e

logger.info("All Verification Steps Passed Successfully!")
