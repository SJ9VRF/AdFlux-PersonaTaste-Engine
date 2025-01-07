from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration

def download_bert(save_path="models/foundation_models/bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"BERT model saved to {save_path}")

def download_gpt(save_path="models/foundation_models/gpt2"):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"GPT model saved to {save_path}")

def download_t5(save_path="models/foundation_models/t5-small"):
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"T5 model saved to {save_path}")

if __name__ == "__main__":
    download_bert()
    download_gpt()
    download_t5()

