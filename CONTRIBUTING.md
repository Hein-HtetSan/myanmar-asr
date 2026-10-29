# Contributing

Contributions that improve reproducibility, Burmese text handling, model evaluation, or deployment are welcome.

1. Open an issue describing the problem, relevant dataset split, model, and environment. Do not attach private audio, credentials, or model weights you cannot redistribute.
2. Keep changes focused and document any change to preprocessing or metric calculation. Report WER and CER with the exact split, dataset version, model checkpoint, and library versions.
3. Run `python -m compileall -q scripts` and `docker compose -f services/docker-compose.yml config --quiet` (with a configured `.env`) for changes that touch those paths.
4. Submit a pull request describing the change, evidence, and any hardware-dependent checks that were not run.

Do not commit `.env`, Hugging Face tokens, API keys, raw audio, or generated model checkpoints. If you find a security issue, contact the maintainer privately through GitHub rather than posting secrets in a public issue.
