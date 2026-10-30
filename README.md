# ATA Recoder DAPT20 Seed 42

Single-seed productionization package for the validated DAPT20 hierarchical ATA classifier. The package is designed for **shadow scoring first**; it does not replace the legacy model's Mongo field and never enables automatic recoding.

## Model and evaluation

- Model: DAPT20 DeBERTa hierarchical classifier, seed 42.
- Test-set result: 70.36% top-1 and 92.56% recall@5 on 18,385 held-out rows.
- Latency reference: about 20.5 ms/item on H100, using the benchmark's three-record payloads and seven repeats.
- Inference: max length 256, stride 128, average window probabilities, then rank ATA codes with `P(code) * P(chapter)^1.0`.
- Checkpoint weights are large and must be obtained from the Azure ML artifact store. They are intentionally excluded from Git.

The legacy production model's old top-5 score is not the 92.44% corrected-baseline score: a fresh run of the bundled production model on the same held-out split scored 63.07% top-1 / 84.99% top-5. The seed-42 gain is therefore materially larger than an earlier baseline comparison suggested.

## Artifact layout

Download/copy the seed 42 artifact directory and DAPT encoder config into a deployment-accessible Azure ML datastore or mounted folder. The seed directory must include:

```text
models/seed42/
  best_model/model.safetensors
  best_model/tokenizer.json
  best_model/tokenizer_config.json
  best_model/vocab.json
  best_model/merges.txt
  label_maps.json
```

The encoder config directory must contain the DAPT encoder `config.json`. The full encoder weights are inside the seed-42 `model.safetensors`; the config is used to instantiate the matching architecture.

Reference artifact locations are recorded in [model-artifacts.json](model-artifacts.json). Never commit model files, database credentials, or credential-bearing connection strings.

## Local checks

The pure ranking helpers use only NumPy and can be tested without loading the model:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
```

For model inference, install dependencies from `requirements.txt` in the serving image/environment.

## Shadow scoring

The Mongo runner defaults to read-only scoring. It writes only when `--write` is explicitly supplied, and its destination defaults to the separate `ataRecoderMLSeed42` field. The legacy `ataRecoderML` field is never modified. The stored document mirrors the legacy top-five shape and sets `autoRecode=false`, `qaReview=true`. Read-only runs can write a privacy-minimized JSONL of record IDs and predictions with `--output-jsonl`; raw report text is not emitted.

```bash
export MONGODB_URI='mongodb+srv://...'
ata-recorder-shadow \
  --model-dir /mnt/models/seed42 \
  --encoder-config-dir /mnt/models/dapt_encoder \
  --days 10 --limit 100 --device cuda
```

The command above is read-only. After reviewing the output and granting explicit write approval, add `--write` to store the shadow field. Keep writes bounded with `--limit` during canary validation.

The current Azure template reads the existing password-only Mongo secret file and builds the connection string using the legacy application user/host defaults. For other environments, set `MONGODB_USER` and `MONGODB_HOST`, or provide a full URI file instead.

### Input-text parity gate

Training/evaluation used `reported_by_clean`; the legacy Mongo scorer reads `reason` and performs legacy cleanup/abbreviation expansion. This package deliberately does not inherit that legacy expansion. Before production writes, confirm the Mongo `reason` value has the same cleaning/content as `reported_by_clean`, or add and validate an approved DAPT-specific cleaner using paired records. Until that is established, use offline/shadow comparison only and do not enable automatic recoding.

## Azure ML

[`azureml_shadow_job.yml`](azureml_shadow_job.yml) is a manual, bounded, **read-only by default** command-job template. Supply model artifact folders and the Mongo connection secret as job inputs via approved secure storage. It scores at most 100 records unless changed. Add `--write` only after a shadow review and security approval.

## Deployment gates

1. Verify model/tokenizer/label-map artifact checksums and load the checkpoint on the target compute.
2. Verify production-text parity against `reported_by_clean` on a reviewed sample.
3. Run read-only shadow scoring, then write only to `ataRecoderMLSeed42` for a bounded canary.
4. Compare coverage, top-five overlap, latency, failures, and confidence calibration against adjudicated outcomes.
5. Keep `autoRecode=false` until a separate confidence calibration study approves a threshold.
6. Preserve legacy rollback; do not overwrite the old model bundle or Mongo field during evaluation.
