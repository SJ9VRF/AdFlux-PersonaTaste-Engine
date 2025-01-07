from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import DatasetDict

def train_custom_model(dataset, model_name, num_labels, save_path, task_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Tokenize dataset
    def tokenize_data(example):
        return tokenizer(example["text"], padding=True, truncation=True)

    tokenized_dataset = dataset.map(tokenize_data, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"training_output/{task_name}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        save_strategy="epoch",
        logging_dir=f"logs/{task_name}",
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )

    # Train and save model
    trainer.train()
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Custom model for {task_name} saved to {save_path}")

if __name__ == "__main__":
    from datasets import load_dataset

    # Load synthetic dataset
    dataset = DatasetDict.load_from_disk("data/processed/synthetic_dataset")

    # Train models
    train_custom_model(dataset, "bert-base-uncased", num_labels=4, save_path="models/custom_models/next_click_predictor", task_name="Next Click Predictor")
    train_custom_model(dataset, "bert-base-uncased", num_labels=2, save_path="models/custom_models/ad_ctr_predictor", task_name="Ad CTR Predictor")
    train_custom_model(dataset, "bert-base-uncased", num_labels=2, save_path="models/custom_models/purchase_likelihood", task_name="Purchase Likelihood Model")

