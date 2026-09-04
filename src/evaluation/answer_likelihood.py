import torch


def answer_nll(model, tokenizer, device, question, answer):
    """
    Calculate negative log-likelihood of the answer
    conditioned on the question.

    Lower NLL = stronger model knowledge.
    Higher NLL = weaker model knowledge.
    """

    prompt = f"Question: {question}\nAnswer:"

    prompt_tokens = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    answer_tokens = tokenizer(
        answer,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    input_ids = torch.cat(
        [
            prompt_tokens["input_ids"],
            answer_tokens["input_ids"]
        ],
        dim=1
    ).to(device)

    attention_mask = torch.ones_like(input_ids).to(device)

    # Ignore the prompt tokens when calculating loss.
    labels = input_ids.clone()

    prompt_length = prompt_tokens["input_ids"].shape[1]

    labels[:, :prompt_length] = -100

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    return outputs.loss.item()