# Repository checks

From the repository root, with SoundFile installed:

```sh
python -m unittest discover -s tests -v
```

Checks cover catalog curation, download pairing, clean notebook syntax, Whisper Small full/quick split selection, normalized-text leakage, and checkpoint archive recovery. Model integration checks require the [training dependencies](../notebook/requirements-training.txt) and skip when unavailable. Tiny random model checks verify mechanics, not quality or GPU capacity.

No test launches full Whisper Small fine-tuning. Run the notebook's quick mode on the target GPU before a full experiment. Audio, caches and checkpoints stay local and ignored by Git.
