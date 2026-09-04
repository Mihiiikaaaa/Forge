from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def main():
    print("Loading tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    print("\nModel loaded successfully!")
    print(f"Model parameters: {model.num_parameters():,}")

    prompt = "Who is Jaime Vasquez?"

    print(f"\nPrompt: {prompt}")

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print("\nModel response:")
    print(response)


if __name__ == "__main__":
    main()