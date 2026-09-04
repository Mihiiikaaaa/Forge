from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import torch


MODEL_PATH = "models/tofu_baseline"
FORGET_PATH = "data/processed/forget_1"
RETAIN_PATH = "data/processed/retain_1"

NUM_EXAMPLES = 10


def generate_response(model, tokenizer, question):

    prompt = (
        "Question: "
        + question
        + "\nAnswer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return response


def evaluate(dataset, model, tokenizer, name):

    print("\n===================================")
    print(name)
    print("===================================")

    for i in range(
        min(NUM_EXAMPLES, len(dataset))
    ):

        question = dataset[i]["question"]
        expected = dataset[i]["answer"]

        response = generate_response(
            model,
            tokenizer,
            question
        )

        print(f"\nExample {i + 1}")

        print("QUESTION:")
        print(question)

        print("\nEXPECTED:")
        print(expected)

        print("\nMODEL:")
        print(response)


def main():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    print("Loading trained model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    forget = load_from_disk(
        FORGET_PATH
    )

    retain = load_from_disk(
        RETAIN_PATH
    )

    evaluate(
        forget,
        model,
        tokenizer,
        "FORGET SET"
    )

    evaluate(
        retain,
        model,
        tokenizer,
        "RETAIN SET"
    )


if __name__ == "__main__":
    main()