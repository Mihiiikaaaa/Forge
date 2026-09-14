# Unlearning baselines

`run_unlearning.py` always loads `models/tofu_baseline` afresh and saves a
separate Hugging Face model directory. It never uses the baseline directory as
an output. All methods train on answer tokens only: question/prompt tokens are
masked from causal-LM likelihood calculations.

Gradient Ascent (GA) minimises the negative forget-answer cross-entropy, which
maximises loss on the forget examples. NPO compares forget-answer log likelihood
with a frozen copy of the baseline and minimises
`softplus(beta * (log p_current - log p_reference))`; this pushes the current
likelihood below the reference likelihood. RMU is a lightweight RMU-style
approximation, not a claim of an exact paper implementation: it drives the
final non-padding representation of each forget example at `target_layer` to a
fixed seed-42 random vector, while an MSE penalty keeps retain representations
near a frozen baseline reference. `target-coefficient`, `retain-coefficient`,
and `target-norm` control that objective.

Run from the repository root, for example:

```powershell
python src/unlearning/run_unlearning.py --method ga --epochs 1 --learning-rate 1e-5
python src/unlearning/run_unlearning.py --method npo --beta 0.1 --epochs 1
python src/unlearning/run_unlearning.py --method rmu --target-layer -1 --retain-coefficient 1.0 --epochs 1
```

Optional YAML keys match the CLI names (`learning_rate`, `target_layer`, etc.):
`--config config/unlearning.yaml`. CLI values override the YAML file. The output
defaults are `models/unlearned/ga`, `models/unlearned/npo`, and
`models/unlearned/rmu`; each includes `unlearning_config.json` and
`training_history.json` in addition to model/tokenizer files.
