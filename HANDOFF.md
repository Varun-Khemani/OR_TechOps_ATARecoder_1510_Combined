# KPaaS Handoff: DAPT20 Seed 42

This handoff is for the engineer integrating `dapt20-seed42-v1` into the existing KPaaS deployment. The GitHub project is private and currently owned by `Varun-Khemani` because the authenticated account was denied `AAInternal`'s repository-creation permission.

## Access

Repository: https://github.com/Varun-Khemani/OR_TechOps_ATARecoder_1510_Combined

The owner must add the integrating engineer as a repository collaborator, or an `AAInternal` organization owner can transfer the repository into the organization. Do not send credentials, Mongo secrets, or Azure tokens in chat or commit them to Git.

## What is in this repo

- `src/ata_recoder_seed42/inference.py`: seed-42 model loading and inference.
- `src/ata_recoder_seed42/mongo_shadow.py`: bounded Mongo scorer; read-only unless `--write` is specified.
- `azureml_shadow_job.yml`: Azure ML H100, 100-record, read-only shadow template.
- `model-artifacts.json`: model/encoder artifact references and decoding configuration.
- `README.md`: installation, run commands, safety behavior, and validation gates.

The 1.6 GB weights are not in Git. Pull them from Azure ML using the artifact location in `model-artifacts.json`. The seed-42 folder must contain `best_model/model.safetensors`, tokenizer files, and `label_maps.json`. The encoder-config folder supplies the DeBERTa architecture config; the complete encoder weights are in the seed checkpoint.

## KPaaS integration tasks

1. Add this private repository as a source dependency or mirror it into the approved `AAInternal` organization repo after an org owner grants the required permission.
2. Add the package dependencies from `requirements.txt` to the KPaaS image/runtime. Confirm the runtime supports PyTorch 2.7.1, Transformers 4.45.2, and the target CUDA/runtime combination; CPU fallback is available but substantially slower.
3. Mount/download the seed-42 model directory and encoder config from Azure ML into the container. Do not put either large model files or credentials in the Git image context.
4. Provide Mongo access through KPaaS/Key Vault/AML secret injection. The sample Azure job references the existing development password file and is for controlled shadow validation only; replace it with the approved KPaaS secret mechanism and production connection settings.
5. Integrate the scorer into the existing scheduled/batch lifecycle. Preserve the current eligibility filter and existing `ataRecoderML` output. During shadow mode, write only to `ataRecoderMLSeed42` or emit a separate prediction file/topic.
6. Run the package tests, then a read-only bounded shadow pass. Compare outputs and latency against adjudicated records and the existing model before any write-enabled canary.
7. Keep `autoRecode=false` and `qaReview=true`. A separate calibration/approval step is required before automatic recoding can be considered.

## Model contract

- Input training field: `reported_by_clean`.
- Current Mongo runner field: `reason`.
- Max sequence length: 256; stride: 128.
- Long-input window aggregation: mean of each window's class probabilities.
- ATA score ordering: `P(ATA code) * P(chapter)^1.0`.
- Test result: 70.36% top-1 / 92.56% recall@5 on the 18,385-row held-out benchmark.
- H100 latency reference: about 20.5 ms per record on the reduced three-record payload benchmark; production throughput depends on KPaaS CPU/GPU, batching, Mongo/network time, and record length.

### Blocking parity check

Training/evaluation consumed `reported_by_clean`; the old production scorer consumes Mongo `reason` and applies legacy cleanup plus abbreviation expansion. This DAPT package does not apply the legacy abbreviation expansion. Before writing shadow output, confirm the Mongo `reason` field matches `reported_by_clean` closely enough for this model or implement and validate a DAPT-approved transformation. The current Azure shadow YAML uses `reason` only as a testable placeholder and must not be treated as production-approved without that check.

### Confidence and automatic action

The output score is the four-digit-head probability attached to a ranking produced with the chapter prior. It is not calibrated for the legacy `0.9` auto-recode threshold. Do not use that threshold for DAPT seed 42. Automatic recoding must remain disabled until a separately reviewed calibration/threshold study is complete.

## Local smoke test

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
```

A read-only bounded run against an approved non-production Mongo connection:

```bash
ata-recorder-shadow \
  --model-dir /mnt/models/seed42 \
  --encoder-config-dir /mnt/models/dapt_encoder \
  --mongo-uri-file /run/secrets/mongodb_uri \
  --days 10 --limit 100 --device cuda \
  --output-jsonl /tmp/seed42-shadow.jsonl
```

This does not write to Mongo. Only add `--write` after the text-parity, security, and shadow-review gates are approved; even then, keep it pointed at the separate `ataRecoderMLSeed42` field.

## Rollback

The DAPT scorer is additive during shadow/canary. Rollback by disabling its KPaaS scheduled task/route and ignoring the `ataRecoderMLSeed42` shadow field. Do not modify or delete the current `ataRecoderML` data or legacy deployment bundle during this evaluation.
