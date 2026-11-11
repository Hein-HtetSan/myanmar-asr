# Research scripts

Run commands from the repository root. Each directory documents its inputs and environment:

- [data](data/README.md): downloads, curation, combined catalogs and exports.
- [training](training/README.md): historical trainers; use [Whisper Small](../notebook/whisper-small/README.md) for the current combined-catalog recipe.
- [evaluation](evaluation/README.md): scoring and dataset analysis.
- [deploy](deploy/README.md): inference API and demo.
- [infra](infra/README.md): GPU and infrastructure helpers.
- [utils](utils/README.md): historical maintenance tools.
- [viz](viz/README.md): metric exports and figures.

Some historical scripts execute on import or contain environment-specific paths. Read the relevant guide first. See [reproducibility](../docs/REPRODUCIBILITY.md) before comparing results.
