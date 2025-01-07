from transformers import T5ForConditionalGeneration, T5Tokenizer

def search_to_purchase_mapping(search_query, model_path="models/foundation_models/t5-small"):
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    input_text = f"search to purchase: {search_query}"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_beams=5, early_stopping=True)
    purchase_suggestion = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Suggested purchase: {purchase_suggestion}")

if __name__ == "__main__":
    search_to_purchase_mapping("Buy headphones with noise cancellation")

