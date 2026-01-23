
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
    evaluation_strategy="no",
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
    evaluation_strategy="no",
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
