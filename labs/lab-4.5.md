# Lab 4.5: Local Training Job Monitoring

In this lab, you will run a complete local training workflow that mirrors a managed training job, without AWS Console or S3.

## Learning Objectives

By the end of this lab, you will be able to:

- Launch a local Hugging Face training job
- Track training metrics in real time
- Use local dataset paths instead of remote input channels
- Analyze post-training metrics and identify optimization opportunities

## Prerequisites

Use the repository standard setup:

```bash
uv sync --all-extras
```

## Lab Exercises

### Exercise 1: Run a Local Training Job

Run from the repository root:

```bash
uv run python examples/training/local_job.py \
  --data examples/loading/status.csv \
  --output-dir examples/training/trail_classifier_local \
  --log-file examples/training/trail_training_log.jsonl
```

What this run includes:
- Local CSV input dataset
- Trainer API execution
- Metric collection for `loss`, `eval_loss`, `eval_accuracy`, `eval_precision`, `eval_recall`, `eval_f1`
- Model artifact output in `examples/training/trail_classifier_local`

### Exercise 2: Monitor Metrics During Training

In a second terminal:

```bash
tail -f examples/training/trail_training_log.jsonl
```

Watch:
- `loss` and `eval_loss` trends
- `eval_accuracy` and `eval_f1` stability
- Learning-rate behavior across steps

### Exercise 3: Analyze Training Results

After the run ends:

```bash
uv run python examples/training/analyze_training_log.py \
  --log-file examples/training/trail_training_log.jsonl
```

Use the summary to determine:
- Best epoch metrics
- Signals of overfitting risk
- Next optimization actions (learning rate, epochs, fp16)

### Exercise 4: Validate with Inference

```bash
uv run python examples/inferencing/inference.py examples/training/trail_classifier_local
```

## Challenge

1. Run two experiments with different learning rates (`2e-5` and `1e-5`).
2. Compare `eval_f1` and `eval_loss` in the resulting logs.
3. Select the better configuration and justify your choice.
