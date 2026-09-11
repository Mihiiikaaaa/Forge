# FORGE Evaluation Protocol

## 1. Forgetting Effectiveness

FORGE evaluates whether targeted knowledge is reduced after unlearning.

Metrics:

- Answer probability
- Relative likelihood drop
- Truth Ratio
- Forget Quality
- ROUGE-L where applicable

Higher forgetting effectiveness means stronger suppression/removal of the targeted knowledge.

## 2. Retention

Retention measures whether non-target knowledge remains usable after unlearning.

Metrics:

- Retained answer likelihood
- Model Utility
- ROUGE-L where applicable

Higher retention indicates lower collateral damage.

## 3. Robustness

FORGE evaluates forgetting beyond the original training question.

Robustness tests include:

1. Direct recall
2. Paraphrased recall
3. Indirect inference
4. Relearning resistance

## 4. Collateral Damage

FORGE measures knowledge degradation at different semantic distances from the forget target.

Knowledge is divided into:

- Related
- Moderately related
- Unrelated

The purpose is to measure the blast radius of targeted unlearning.

## 5. Experimental Variables

Experiments will be evaluated using:

- 1% forgetting
- 5% forgetting
- 10% forgetting
- Multiple unlearning baselines
- FORGE method

## 6. Research Principle

A successful unlearning method should:

- Forget the target knowledge.
- Resist paraphrased and indirect recovery.
- Resist relearning.
- Preserve unrelated knowledge.
- Minimize collateral damage around the target.
