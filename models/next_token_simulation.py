from transformers import AutoModelForCausalLM, AutoTokenizer

def simulate_next_token(sequence, model_path="models/foundation_models/gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    inputs = tokenizer.encode(sequence, return_tensors="pt")
    outputs = model.generate(inputs, max_length=len(inputs[0]) + 1, do_sample=True, top_k=50, top_p=0.95)
    next_token = tokenizer.decode(outputs[0], skip_special_tokens=True).split()[-1]
    
    print(f"Next token predicted: {next_token}")

if __name__ == "__main__":
    simulate_next_token("User clicked on a product and")

