from transformers import AutoModelForSequenceClassification, AutoTokenizer

def run_simulation(config_path):
    import json

    with open(config_path, 'r') as f:
        config = json.load(f)

    model_name = config["model_name"]
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    user_sequence = config["user_sequence"]
    inputs = tokenizer(user_sequence, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)
    
    print("Predicted user actions:", predictions)
