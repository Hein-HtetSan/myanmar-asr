import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'notebook'))
NOTEBOOKS = [ROOT/'notebook'/name/'train.ipynb' for name in ('whisper-turbo','dolphin','seamless')]
DEPENDENCIES = all(importlib.util.find_spec(name) for name in
                   ('torch','transformers','mlflow','jiwer','datasets','accelerate','soundfile'))


class NotebookStructureTest(unittest.TestCase):
    def test_three_clean_notebooks_compile(self):
        for path in NOTEBOOKS:
            with self.subTest(path=path):
                notebook = json.loads(path.read_text())
                self.assertEqual(notebook['nbformat'], 4)
                code = []
                for index, cell in enumerate(notebook['cells']):
                    if cell['cell_type'] != 'code':
                        continue
                    self.assertFalse(cell['outputs'])
                    self.assertIsNone(cell['execution_count'])
                    source = ''.join(cell['source'])
                    sanitized = '\n'.join('pass' if line.lstrip().startswith('%') else line
                                          for line in source.splitlines())
                    compile(sanitized, f'{path}:cell-{index}', 'exec')
                    code.append(source)
                combined = '\n'.join(code)
                self.assertIn('QUICK_RUN = True', combined)
                self.assertIn('tracked_run(', combined)
                self.assertIn('latest_checkpoint(', combined)
                self.assertIn('include_labels=False', combined)
                self.assertNotIn('/workspace/', combined)
                self.assertNotIn('/Users/', combined)


@unittest.skipUnless(DEPENDENCIES, 'Install the notebook training dependencies to run model checks')
class TrainingHelpersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import training_common
        cls.common = training_common

    def test_scoring_separates_spacing_from_raw_errors(self):
        scores = self.common.score_texts(['က ခ'], ['ကခ'])
        self.assertEqual(scores['cer_no_space'], 0)
        self.assertGreater(scores['cer_raw'], 0)
        self.assertGreater(scores['wer'], 0)

    def test_decoder_start_is_removed_and_overlong_labels_are_not_truncated(self):
        from types import SimpleNamespace
        processor = SimpleNamespace(tokenizer=lambda text: SimpleNamespace(input_ids=[7, 4, 5, 2]))
        self.assertEqual(self.common.label_ids(processor, 'text', 'whisper', 7), [4, 5, 2])
        with self.assertRaisesRegex(ValueError, 'No labels fit'):
            self.common.filter_labels([{'text':'text','sample_id':'a'}], processor, 'whisper', 7, 3)

    def test_seamless_masks_and_shifted_decoder_inputs(self):
        import numpy as np
        collate = self.common.SpeechCollator('seamless', 0, 3)
        features = [dict(input_features=np.ones((3, 4)), attention_mask=np.ones(3), labels=[4, 3]),
                    dict(input_features=np.ones((5, 4)), attention_mask=np.ones(5), labels=[5, 6, 3])]
        batch = collate(features)
        self.assertEqual(tuple(batch['input_features'].shape), (2, 5, 4))
        self.assertEqual(batch['attention_mask'][0].tolist(), [1,1,1,0,0])
        self.assertEqual(batch['labels'][0].tolist(), [4,3,-100])
        self.assertEqual(batch['decoder_input_ids'].tolist(), [[3,4,3],[3,5,6]])
        unlabeled = collate([{k:v for k,v in features[0].items() if k!='labels'}])
        self.assertNotIn('labels', unlabeled)
        self.assertNotIn('decoder_input_ids', unlabeled)

    def test_tiny_whisper_and_seamless_training_tracking_and_resume_checks(self):
        import mlflow
        import numpy as np
        import torch
        from transformers import (WhisperConfig, WhisperForConditionalGeneration,
                                  SeamlessM4Tv2Config, SeamlessM4Tv2ForSpeechToText,
                                  Seq2SeqTrainingArguments, Seq2SeqTrainer)
        torch.set_num_threads(1)
        uri = os.environ.get('ASR_TEST_MLFLOW_URI')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            uri = uri or (root/'mlruns').as_uri()
            for family in ('whisper','seamless'):
                with self.subTest(family=family):
                    run_dir = root/family
                    if family == 'whisper':
                        config = WhisperConfig(vocab_size=16, num_mel_bins=4, d_model=16,
                            encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                            decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
                            max_source_positions=10, max_target_positions=12,
                            pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1,
                            suppress_tokens=[], begin_suppress_tokens=[])
                        model = WhisperForConditionalGeneration(config)
                        features = np.zeros((4,20), dtype=np.float32)
                        generation = {}
                    else:
                        config = SeamlessM4Tv2Config(vocab_size=16, hidden_size=16,
                            encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                            decoder_attention_heads=2, encoder_ffn_dim=32, decoder_ffn_dim=32,
                            speech_encoder_layers=1, speech_encoder_attention_heads=2,
                            speech_encoder_intermediate_size=32, feature_projection_input_dim=4,
                            conv_depthwise_kernel_size=3, adaptor_kernel_size=4, adaptor_stride=2,
                            max_position_embeddings=32, pad_token_id=0, bos_token_id=2,
                            eos_token_id=3, decoder_start_token_id=3, speech_encoder_layerdrop=0.0)
                        model = SeamlessM4Tv2ForSpeechToText(config)
                        model.generation_config.text_decoder_lang_to_code_id = {'mya':4}
                        features = np.zeros((20,4), dtype=np.float32)
                        generation = {'tgt_lang':'mya'}
                    encoder = model.model.encoder if family == 'whisper' else model.speech_encoder
                    for parameter in encoder.parameters():
                        parameter.requires_grad = False
                    model.config.use_cache = False
                    model.generation_config.max_length = 8
                    model.generation_config.max_new_tokens = None
                    collator = self.common.SpeechCollator(family, 0, config.decoder_start_token_id)
                    sample = {'input_features':features, 'attention_mask':np.ones(20), 'labels':[4,5,config.eos_token_id]}
                    args = Seq2SeqTrainingArguments(output_dir=str(run_dir), use_cpu=True,
                        max_steps=1, per_device_train_batch_size=1, per_device_eval_batch_size=1,
                        eval_strategy='steps', eval_steps=1, save_steps=1, logging_steps=1,
                        predict_with_generate=True, generation_max_length=8,
                        load_best_model_at_end=True, metric_for_best_model='cer_no_space',
                        greater_is_better=False, label_smoothing_factor=0.1,
                        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant':False},
                        report_to=[], disable_tqdm=True, dataloader_pin_memory=False)
                    trainer = self.common.SpeechTrainer(model=model, args=args,
                        train_dataset=[sample], eval_dataset=[sample], data_collator=collator,
                        generation_kwargs=generation,
                        compute_metrics=lambda result: {'cer_no_space':1.,'cer_raw':2.,'wer':3.},
                        callbacks=[self.common.MLflowMetricsCallback()])
                    configuration = {'family':family,'quick_run':True,'fixture':'tiny-random-v1'}
                    with self.common.tracked_run(run_dir, uri, 'notebook-verification', family, configuration) as active:
                        run_id = active.info.run_id
                        trainer.train()
                        unlabeled = {k:v for k,v in sample.items() if k!='labels'}
                        result = trainer.predict([unlabeled])
                        self.assertEqual(result.predictions.shape[0], 1)
                    saved = mlflow.MlflowClient().get_run(run_id)
                    self.assertEqual(saved.info.status, 'FINISHED')
                    self.assertEqual(saved.data.metrics['eval_cer_no_space'], 1.)
                    client = mlflow.MlflowClient()
                    config_artifact = client.download_artifacts(run_id, 'configuration/run.json', str(root))
                    self.assertEqual(json.loads(Path(config_artifact).read_text()), configuration)
                    checkpoint = self.common.latest_checkpoint(run_dir)
                    self.assertEqual(checkpoint.name, 'checkpoint-1')
                    # A truncated newer optimizer must not hide the usable older save.
                    import shutil
                    corrupt = run_dir/'checkpoint-2'
                    shutil.copytree(checkpoint, corrupt)
                    state_path = corrupt/'trainer_state.json'
                    state = json.loads(state_path.read_text()); state['global_step'] = 2
                    state_path.write_text(json.dumps(state))
                    (corrupt/'optimizer.pt').write_bytes(b'PK\x03\x04truncated')
                    self.assertEqual(self.common.latest_checkpoint(run_dir), checkpoint)
                    with self.common.tracked_run(run_dir, uri, 'notebook-verification', family, configuration) as resumed:
                        self.assertEqual(resumed.info.run_id, run_id)
                    with self.assertRaisesRegex(ValueError, 'settings/data changed'):
                        with self.common.tracked_run(run_dir, uri, 'notebook-verification', family,
                                                     {**configuration,'fixture':'changed'}):
                            pass
                    with patch.object(Seq2SeqTrainer, 'evaluate', return_value={}) as evaluate:
                        trainer.evaluate()
                        if family == 'seamless':
                            self.assertEqual(evaluate.call_args.kwargs['tgt_lang'], 'mya')


if __name__ == '__main__':
    unittest.main()
