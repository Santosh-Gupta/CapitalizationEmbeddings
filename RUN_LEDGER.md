# Run Ledger

This file records launched GPU/CPU experiment batches so work is not duplicated
after context compression.

## RunPod

```text
host: root@203.57.40.173 -p 10267
current host after redeploy: root@203.57.40.158 -p 10019
repo: /workspace/repos/CapitalizationEmbeddings
checkpoint root: /workspace/capitalization_embeddings/checkpoints
log root: /workspace/capitalization_embeddings/logs
reports root: /workspace/capitalization_embeddings/reports
```

Before launching work:

1. Confirm GPU/process state with `nvidia-smi` and `pgrep`.
2. Add or update an entry in this file with command, seeds, roots, and log path.
3. Use `resume_benchmark_sweep.py` so completed model/seed rows are skipped.
4. After completion, update this file and `PAPER_EVIDENCE_STATUS.md`.

## Completed Batches

### Mixed-Case Required Token 5-Seed Completion

Status: completed.

Result roots:

```text
capitalized current-best:
/workspace/capitalization_embeddings/checkpoints/mixed_case_eval_3seed

matched cased/uncased OntoNotes/PTB:
/workspace/capitalization_embeddings/checkpoints/required_token_baselines_3seed
```

Seeds:

```text
13 21 34 55 89
```

Tasks:

```text
conll2003_ner
wnut17_ner
ontonotes5_ner
ptb_pos
```

Notes:

- Historical root names include `3seed`, but these roots now contain 5 seeds for
  the rows described above.
- CoNLL/WNUT cased and uncased controls are in
  `/workspace/capitalization_embeddings/checkpoints/significance_5seed`.

### Mixed-Case Selected Sequence 5-Seed Completion

Status: completed.

Result roots:

```text
capitalized current-best:
/workspace/capitalization_embeddings/checkpoints/mixed_case_sequence_5seed

matched cased/uncased controls:
/workspace/capitalization_embeddings/checkpoints/significance_5seed
```

Seeds:

```text
13 21 34 55 89
```

Tasks:

```text
tweet_eval_irony
tweet_eval_offensive
sst5
twenty_newsgroups
```

## Active Or Next Batch

### Case-Channel Ablation Batch

Status: planned; script prepared locally, not yet launched.

Purpose:

- Test whether the current best result depends on the mixed-case state,
  capitalization embedding dropout, and auxiliary capitalization loss.
- Keep the first resumed GPU session focused on CoNLL-2003 and Kaggle/Walia NER
  before spending on wider seed expansion.

Prepared launcher:

```bash
cd /workspace/repos/CapitalizationEmbeddings
bash scripts/run_case_ablation_batch.sh
```

The launcher trains missing checkpoints only:

```text
4-state no-dropout:
/workspace/capitalization_embeddings/checkpoints/mlm/mixed_case_no_dropout/capitalized_from_3class_steps3000_lr2e5/final

4-state no-aux-loss:
/workspace/capitalization_embeddings/checkpoints/mlm/mixed_case_no_aux_loss/capitalized_from_3class_steps3000_lr2e5_drop01/final
```

It then evaluates these variants:

```text
three_class_existing
four_class_dropout_current_best
four_class_no_dropout
four_class_no_aux_loss
```

Tasks and seeds:

```text
tasks: conll2003_ner, kaggle_walia_ner
seeds: 13 21 34 55 89
```

Result root:

```text
/workspace/capitalization_embeddings/checkpoints/ablations_case_channel_5seed
```

Log root:

```text
/workspace/capitalization_embeddings/logs/ablations_case_channel
```

Before launch:

1. Pull latest `main`.
2. Confirm dependencies are installed with `pip install -e .`.
3. Confirm GPU idle with `nvidia-smi`.
4. Run `bash -n scripts/run_case_ablation_batch.sh`.

### Paper Final High-Variance 20-Seed Expansion

Status: stopped early; partial results retained.

Started: 2026-05-14 UTC.

RunPod PID:

```text
60740 (exited)
```

Purpose:

- Expand WNUT-17 and CoNLL-2003 from 5 to 20 seeds for all three matched model
  families.
- Expand selected uncased-favored sequence tasks from 5 to 20 seeds for all
  three matched model families.
- This is the main remaining GPU-heavy batch before deciding whether the claim
  is statistically strong enough or must be narrowed.

Additional seeds:

```text
144 233 377 610 987 1597 2584 4181 6765 10946 17711 28657 46368 75025 121393
```

Checkpoints:

```text
capitalized:
/workspace/capitalization_embeddings/checkpoints/mlm/mixed_case_dropout/capitalized_from_3class_steps3000_lr2e5_drop01/final

uncased:
/workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/uncased_from_task_mix_steps3000_lr2e5/final

cased:
/workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/cased_from_task_mix_steps3000_lr2e5/final
```

Result roots:

```text
capitalized token:
/workspace/capitalization_embeddings/checkpoints/mixed_case_eval_3seed

capitalized sequence:
/workspace/capitalization_embeddings/checkpoints/mixed_case_sequence_5seed

cased/uncased token and sequence:
/workspace/capitalization_embeddings/checkpoints/significance_5seed
```

Launcher/log:

```text
/workspace/capitalization_embeddings/logs/run_paper_final_20seed_expansion.sh
/workspace/capitalization_embeddings/logs/paper_final_20seed_expansion.log
```

Tasks:

```text
token:
conll2003_ner
wnut17_ner

sequence:
tweet_eval_irony
tweet_eval_offensive
sst5
twenty_newsgroups
```

Partial completion observed on 2026-05-14 UTC:

```text
/workspace/capitalization_embeddings/checkpoints/mixed_case_eval_3seed
  conll2003_ner capitalized_pretrained: 20 seeds
  wnut17_ner capitalized_pretrained: 20 seeds

/workspace/capitalization_embeddings/checkpoints/significance_5seed
  conll2003_ner uncased_pretrained: 20 seeds
  conll2003_ner cased_pretrained: 20 seeds
  wnut17_ner uncased_pretrained: 13 seeds
  wnut17_ner cased_pretrained: 13 seeds
  selected sequence tasks: 5 seeds/model

/workspace/capitalization_embeddings/checkpoints/mixed_case_sequence_5seed
  selected sequence tasks capitalized_pretrained: 5 seeds
```

### Paper Final Missing 20-Seed Remainder

Status: stopped after WNUT completion; sequence remainder moved to parallel launcher.

Started: 2026-05-14 UTC.

RunPod PID:

```text
69736 (exited after WNUT controls reached 20/20)
```

Purpose:

- Fill only rows missing from the previous expansion, without rerunning completed
  CoNLL or current-best WNUT rows.
- Complete WNUT-17 cased/uncased controls to 20 seeds.
- Complete the selected sequence task current-best capitalized rows to 20 seeds.
- Complete the selected sequence task cased/uncased controls to 20 seeds.

Launcher/log:

```text
/workspace/repos/CapitalizationEmbeddings/scripts/run_paper_final_missing_20seed_remainder.sh
/workspace/capitalization_embeddings/logs/paper_final_missing_20seed_remainder.log
```

Result roots:

```text
/workspace/capitalization_embeddings/checkpoints/significance_5seed
/workspace/capitalization_embeddings/checkpoints/mixed_case_sequence_5seed
```

If WNUT completes and the remaining sequence work is still serial and GPU
utilization remains low, stop the serial launcher after a clean row boundary and
resume sequence work with:

```text
/workspace/repos/CapitalizationEmbeddings/scripts/run_paper_final_sequence_parallel.sh
```

The parallel launcher runs one worker per selected sequence task. Each worker
uses `resume_benchmark_sweep.py`, so partial rows written by the serial launcher
are skipped.

### Paper Final Sequence 20-Seed Parallel Remainder

Status: unknown; SSH endpoint became unreachable during polling.

Started: 2026-05-14 UTC.

RunPod PID:

```text
71455
```

Launcher/log:

```text
/workspace/repos/CapitalizationEmbeddings/scripts/run_paper_final_sequence_parallel.sh
/workspace/capitalization_embeddings/logs/paper_final_sequence_parallel.log
/workspace/capitalization_embeddings/logs/sequence_parallel/*.log
```

Notes:

- One orphaned serial `tweet_eval_irony` child briefly overlapped with the new
  parallel worker and was killed.
- `tweet_eval_irony` results were checked immediately afterward; no duplicate
  `(model_key, seed)` rows were present.
- Last successful poll, before SSH failure, showed GPU at 100% utilization and:

```text
mixed_case_sequence_5seed
  tweet_eval_irony capitalized_pretrained: 20 seeds
  tweet_eval_offensive capitalized_pretrained: 9 seeds
  sst5 capitalized_pretrained: 11 seeds
  twenty_newsgroups capitalized_pretrained: 9 seeds

significance_5seed selected sequence tasks:
  uncased_pretrained/cased_pretrained/capitalized_pretrained: 5 seeds each
```

- The next poll failed with direct TCP `Connection refused`; the old RunPod SSH
  proxy command rejected the key. Check the RunPod UI for whether the pod
  stopped/restarted or whether the SSH endpoint changed.

### Paper Final Sequence 20-Seed Parallel Resume After Redeploy

Status: superseded by foreground/idempotent resume after detached launch
instability and disk-quota cleanup.

Started: 2026-05-14 UTC.

RunPod host:

```text
root@203.57.40.158 -p 10019
```

RunPod PID:

```text
1959
```

Launcher/log:

```text
/workspace/repos/CapitalizationEmbeddings/scripts/run_paper_final_sequence_parallel.sh
/workspace/capitalization_embeddings/logs/paper_final_sequence_parallel_resume_after_deps.log
/workspace/capitalization_embeddings/logs/sequence_parallel/*.log
```

Redeploy state before resume:

```text
mixed_case_sequence_5seed
  tweet_eval_irony capitalized_pretrained: 20 seeds
  tweet_eval_offensive capitalized_pretrained: 13 seeds
  sst5 capitalized_pretrained: 15 seeds
  twenty_newsgroups capitalized_pretrained: 13 seeds

significance_5seed
  tweet_eval_irony uncased_pretrained: 12 seeds
  tweet_eval_irony cased_pretrained: 11 seeds
  selected remaining sequence tasks: 5 seeds/model
```

Notes:

- Fresh container was missing Python dependencies; first relaunch failed with
  `ModuleNotFoundError: transformers`.
- Installed `requirements-colab.txt` and `pip install -e .`, then relaunched.

### Paper Final Foreground Sequence Control Resume

Status: completed.

Started: 2026-05-14 UTC.

RunPod host:

```text
root@203.57.40.158 -p 10019
```

Active processes observed after local SSH stream disconnect and later
parallelization:

```text
resume_benchmark_sweep.py PID 12218
run_sequence_classification_benchmark.py PID 12734/13035 (tweet_eval_offensive)
additional foreground sessions launched for sst5 and twenty_newsgroups
```

Command:

```bash
cd /workspace/repos/CapitalizationEmbeddings
python -u scripts/resume_benchmark_sweep.py \
  --results-root /workspace/capitalization_embeddings/checkpoints/significance_5seed \
  --models uncased_pretrained cased_pretrained \
  --uncased-checkpoint /workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/uncased_from_task_mix_steps3000_lr2e5/final \
  --cased-checkpoint /workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/cased_from_task_mix_steps3000_lr2e5/final \
  --capitalized-checkpoint /workspace/capitalization_embeddings/checkpoints/mlm/mixed_case_dropout/capitalized_from_3class_steps3000_lr2e5_drop01/final \
  --seeds 144 233 377 610 987 1597 2584 4181 6765 10946 17711 28657 46368 75025 121393 \
  --token-tasks \
  --sequence-tasks tweet_eval_offensive sst5 twenty_newsgroups \
  --sequence-epochs 3 \
  --sequence-batch-size 16 \
  --sequence-learning-rate 2e-5 \
  --no-save-model
```

Pre-run blocker and fix:

- Two foreground resume attempts failed with `OSError: [Errno 122] Disk quota
  exceeded` during `evaluate` metric finalization.
- Root cause was disposable Trainer output directories under
  `/workspace/capitalization_embeddings/checkpoints/benchmarks`, about 173 GB.
- The compact result files are under task roots such as
  `/workspace/capitalization_embeddings/checkpoints/significance_5seed/<task>/results.jsonl`;
  those were retained.
- Cleanup command used:

```bash
pkill -f 'run_sequence_classification_benchmark.py' || true
rm -rf /workspace/capitalization_embeddings/checkpoints/benchmarks
mkdir -p /workspace/capitalization_embeddings/checkpoints/benchmarks
rm -rf /workspace/.cache/huggingface/evaluate /root/.cache/huggingface/evaluate /tmp/* || true
```

Important: `/workspace/capitalization_embeddings/checkpoints/benchmarks` is
disposable per-seed Trainer output. Do not rely on it for final metrics.

Counts immediately before this foreground resume:

```text
mixed_case_sequence_5seed
  tweet_eval_irony capitalized_pretrained: 20
  tweet_eval_offensive capitalized_pretrained: 20
  sst5 capitalized_pretrained: 20
  twenty_newsgroups capitalized_pretrained: 20

significance_5seed
  tweet_eval_irony uncased_pretrained: 20
  tweet_eval_irony cased_pretrained: 20
  tweet_eval_offensive uncased_pretrained: 9
  tweet_eval_offensive cased_pretrained: 9
  sst5 uncased_pretrained: 13
  sst5 cased_pretrained: 13
  twenty_newsgroups uncased_pretrained: 10
  twenty_newsgroups cased_pretrained: 9
```

Counts after SSH stream disconnected, while remote process was still running:

```text
tweet_eval_offensive uncased_pretrained: 12
tweet_eval_offensive cased_pretrained: 11
sst5 uncased_pretrained: 13
sst5 cased_pretrained: 13
twenty_newsgroups uncased_pretrained: 10
twenty_newsgroups cased_pretrained: 9
```

Next action:

Final counts:

```text
tweet_eval_offensive uncased_pretrained: 20
tweet_eval_offensive cased_pretrained: 20
sst5 uncased_pretrained: 20
sst5 cased_pretrained: 20
twenty_newsgroups uncased_pretrained: 20
twenty_newsgroups cased_pretrained: 20
```

Duplicate check:

```text
No duplicate (model_key, seed) rows in the three completed selected sequence
control files.
```

Final reports generated:

```text
/workspace/capitalization_embeddings/reports/final_token_20seed_ner_bootstrap_1000.md
/workspace/capitalization_embeddings/reports/final_token_20seed_holm.md
/workspace/capitalization_embeddings/reports/final_sequence_20seed_macro_f1_bootstrap_1000.md
/workspace/capitalization_embeddings/reports/final_sequence_20seed_accuracy_bootstrap_1000.md
/workspace/capitalization_embeddings/reports/final_sequence_20seed_holm.md
```

Cleanup:

```text
Deleted disposable /workspace/capitalization_embeddings/checkpoints/benchmarks
after reports were generated. Compact JSONL metrics and prediction files remain
under their result roots.
```

Final RunPod state:

```text
GPU idle: NVIDIA GeForce RTX 4090, 0 %, 1 MiB / 24564 MiB
No run_sequence_classification, resume_benchmark_sweep, summarize_benchmark_sweep,
or apply_holm processes remain.
```

The GPU pod can be stopped after this point if no further interactive work is
needed.

## 2026-05-16 Case-Channel Ablation Rerun, Batch-64

Reason:

The initial `scripts/run_case_ablation_batch.sh` ablation run used token batch
size 16. On the RunPod RTX 4090, Walia NER used only about 3 GB VRAM and took
roughly 40 minutes per seed. After `three_class_existing/conll2003_ner` reached
5/5 rows and `three_class_existing/kaggle_walia_ner` reached 2/5 rows, the
low-batch wrapper was stopped to avoid spending many more GPU-hours.

To keep the experiment table clean, the faster run writes to a separate result
root rather than mixing batch-16 and batch-64 rows.

Launcher:

```bash
cd /workspace/repos/CapitalizationEmbeddings
bash scripts/run_case_ablation_batch_b64.sh
```

Result root:

```text
/workspace/capitalization_embeddings/checkpoints/ablations_case_channel_5seed_b64
```

Log root:

```text
/workspace/capitalization_embeddings/logs/ablations_case_channel_b64
```

Protocol:

```text
Models: capitalized_pretrained only
Variants:
  - three_class_existing
  - four_class_dropout_current_best
  - four_class_no_dropout
  - four_class_no_aux_loss
Tasks: conll2003_ner, kaggle_walia_ner
Seeds: 13, 21, 34, 55, 89
Epochs: 3
Token batch size: 64
Learning rate: 3e-5
Save models: no
```

Final status:

```text
DONE 2026-05-16T18:50:11Z
GPU idle after completion: 0 %, 2 MiB / 24564 MiB
All 8 result files complete: 4 variants x 2 tasks x 5 seeds
```

Summary report:

```text
reports/case_channel_ablation_b64_5seed.md
```

Mean test F1:

```text
four_class_dropout_current_best
  conll2003_ner:     0.911972 +/- 0.001666
  kaggle_walia_ner:  0.836846 +/- 0.005355

four_class_no_aux_loss
  conll2003_ner:     0.911216 +/- 0.003689
  kaggle_walia_ner:  0.836203 +/- 0.005220

four_class_no_dropout
  conll2003_ner:     0.910332 +/- 0.002034
  kaggle_walia_ner:  0.836242 +/- 0.006351

three_class_existing
  conll2003_ner:     0.909972 +/- 0.001788
  kaggle_walia_ner:  0.835011 +/- 0.006435
```

### Added Cased-Favored Benchmark 5-Seed Run

Status: completed.

Started: 2026-05-14 UTC.

RunPod host:

```text
root@203.57.40.158 -p 10019
```

Background process:

```text
PID 22273
log: /workspace/capitalization_embeddings/logs/added_cased_favored_5seed.log
results root: /workspace/capitalization_embeddings/checkpoints/added_cased_favored_5seed
```

Purpose:

Run the two added user-requested cased-favored benchmarks that had loaders but
no project results yet:

```text
kaggle_walia_ner
isarcasm_eval_en
```

Checkpoint policy:

Use the current best paper method for the capitalized model and matched
continued-pretraining controls for cased/uncased.

```text
capitalized: /workspace/capitalization_embeddings/checkpoints/mlm/mixed_case_dropout/capitalized_from_3class_steps3000_lr2e5_drop01/final
uncased:     /workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/uncased_from_task_mix_steps3000_lr2e5/final
cased:       /workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/cased_from_task_mix_steps3000_lr2e5/final
```

Command:

```bash
cd /workspace/repos/CapitalizationEmbeddings
python -u scripts/resume_benchmark_sweep.py \
  --results-root /workspace/capitalization_embeddings/checkpoints/added_cased_favored_5seed \
  --models uncased_pretrained cased_pretrained capitalized_pretrained \
  --uncased-checkpoint /workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/uncased_from_task_mix_steps3000_lr2e5/final \
  --cased-checkpoint /workspace/capitalization_embeddings/checkpoints/mlm/real_acronym_mix/cased_from_task_mix_steps3000_lr2e5/final \
  --capitalized-checkpoint /workspace/capitalization_embeddings/checkpoints/mlm/mixed_case_dropout/capitalized_from_3class_steps3000_lr2e5_drop01/final \
  --seeds 13 21 34 55 89 \
  --token-tasks kaggle_walia_ner \
  --sequence-tasks isarcasm_eval_en \
  --token-epochs 3 \
  --token-batch-size 16 \
  --token-learning-rate 3e-5 \
  --sequence-epochs 3 \
  --sequence-batch-size 16 \
  --sequence-learning-rate 2e-5 \
  --no-save-model
```

Early observation:

Walia NER is substantially larger than the small sequence diagnostics. The
first model/seed has 7,194 training steps at batch size 16, so the full
5-seed x 3-model Walia block is expected to take hours, not minutes.

Final counts:

```text
kaggle_walia_ner: 15 rows, 5 seeds x 3 models
isarcasm_eval_en: 15 rows, 5 seeds x 3 models
```

Final means:

```text
kaggle_walia_ner entity F1
  capitalized_pretrained: 0.842183 +/- 0.007401
  cased_pretrained:       0.843687 +/- 0.006108
  uncased_pretrained:     0.826976 +/- 0.006008

isarcasm_eval_en macro-F1
  capitalized_pretrained: 0.603190 +/- 0.017442
  cased_pretrained:       0.607535 +/- 0.010922
  uncased_pretrained:     0.607586 +/- 0.017493
```

Bootstrap/Holm reports:

```text
/workspace/capitalization_embeddings/reports/added_cased_favored_5seed_walia_bootstrap_1000.md
/workspace/capitalization_embeddings/reports/added_cased_favored_5seed_isarcasm_bootstrap_1000.md
/workspace/capitalization_embeddings/reports/added_cased_favored_5seed_holm_margin005.md
/workspace/capitalization_embeddings/reports/added_cased_favored_5seed_evidence_status.md
```

Statistical read:

```text
Kaggle/Walia cap > uncased:
  delta +0.015207, bootstrap CI95 [+0.008672, +0.021349],
  raw p = 0, Holm p = 0, label = win.

Kaggle/Walia cap vs cased:
  delta -0.001504, bootstrap CI95 [-0.009226, +0.004954],
  label = tie at a 0.005 practical margin.

iSarcasm cap vs uncased:
  delta -0.004397, bootstrap CI95 [-0.043937, +0.045050],
  label = tie at a 0.005 practical margin.

iSarcasm cap vs cased:
  delta -0.004345, bootstrap CI95 [-0.050490, +0.049391],
  label = tie at a 0.005 practical margin.
```

Cleanup:

```text
Deleted disposable /workspace/capitalization_embeddings/checkpoints/benchmarks
after reports were generated. Compact JSONL metrics and prediction files remain
under /workspace/capitalization_embeddings/checkpoints/added_cased_favored_5seed.
```

Final RunPod state:

```text
GPU idle: NVIDIA GeForce RTX 4090, 0 %, 1 MiB / 24564 MiB
No active benchmark process remains.
```

The GPU pod can be stopped after this point if no further interactive work is
needed.
