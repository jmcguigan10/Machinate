# sms-spam-transformer

This manual pipeline scaffold was created by `macht new pipeline`.

## Metadata

- pipeline_type: nlp
- template: native-python
- pipeline_config: `machinate.toml`

## Native Usage

```bash
macht task list
macht run validate --experiment baseline
macht run train --experiment baseline
```

## Next Steps

1. If you have not analyzed data yet, prefer the dataset-first flow: `macht grab data`, `macht legate report --data`, then `macht collate pipeline --create --report <report.json>`.
2. If you are intentionally working manual-first, run `macht collate pipeline --report <report.json>` later to materialize `dataset_facts.toml`, `model.toml`, and `training.toml`.
3. Replace the starter task functions in `src/<package>/tasks.py` with real project logic or custom blocks only when you need them.
4. Add runtime dependencies to `requirements.txt` or your preferred env manager.
