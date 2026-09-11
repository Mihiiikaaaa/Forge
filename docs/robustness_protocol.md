# FORGE Robustness Evaluation Protocol

## Direct Recall

Evaluate the original forget questions after unlearning.

Goal:

Determine whether the targeted knowledge remains accessible directly.

## Paraphrased Recall

Evaluate human-verified semantically equivalent paraphrases of forget questions.

Each paraphrase must:

- Preserve the original meaning.
- Preserve the intended answer.
- Be semantically equivalent.
- Be human verified.

Goal:

Determine whether the model can recover forgotten knowledge through linguistic variation.

## Indirect Inference

Test whether the forgotten knowledge can still be reconstructed through related information rather than direct questioning.

Goal:

Measure implicit knowledge leakage.

## Relearning Resistance

Attempt to recover forgotten knowledge through controlled relearning.

Goal:

Determine whether unlearning represents robust knowledge removal or merely temporary suppression.

## Overall Robustness

A robust unlearning method should reduce target knowledge across:

Direct recall → Paraphrase → Indirect inference → Relearning
