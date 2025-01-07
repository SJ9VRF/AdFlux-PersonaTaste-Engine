from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

def train_model(dataset, model_name="bert-base-uncased", output_dir="models/custom_models"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Tokenize data
    tokenized_data = tokenizer(dataset['text'], truncation=True, padding=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data['train'],
        eval_dataset=tokenized_data['test']
    )
    trainer.train()
    return model

