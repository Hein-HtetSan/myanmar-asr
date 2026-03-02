from datasets import load_from_disk, Audio

def normalize_openslr(ds):
    def process(example):
        return {
            "audio":      example["audio"],
            "sentence":   example["sentence"],
            "source":     "openslr80",
            "speaker_id": str(example.get("speaker_id", "unknown")),
            "locale":     "my",
        }
    keep_cols = set(ds.column_names) - {"audio"}
    return ds.map(process, remove_columns=list(keep_cols))

def normalize_fleurs(ds):
    def process(example):
        return {
            "audio":      example["audio"],
            "sentence":   example["transcription"],
            "source":     "fleurs",
            "speaker_id": str(example.get("speaker_id", "unknown")),
            "locale":     "my",
        }
    keep_cols = set(ds.column_names) - {"audio"}
    return ds.map(process, remove_columns=list(keep_cols))

def normalize_commonvoice(ds):
    def process(example):
        return {
            "audio":      example["audio"],
            "sentence":   example["sentence"],
            "source":     "commonvoice",
            "speaker_id": example.get("client_id", "unknown")[:8],
            "locale":     "my",
        }
    keep_cols = set(ds.column_names) - {"audio"}
    return ds.map(process, remove_columns=list(keep_cols))