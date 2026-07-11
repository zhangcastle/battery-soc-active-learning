# Repository Architecture

## Design principles

1. Core research logic belongs in `src/battery_benchmark/`.
2. Notebooks are orchestration and analysis interfaces — not reusable code.
3. Scripts are CLI mirrors of notebooks.
4. No hard-coded paths anywhere in the source code.
5. All paths flow through YAML configs resolved at runtime.
6. No dataset files are committed to Git.
7. Checkpoints enable Colab session resume.

## Module map

```
src/battery_benchmark/
├── utils/         Config loading, path resolution, device, env info
├── data/          Dataset adapters, schema, registry, splits, CycleDataset
├── preprocessing/ Normalisation, sequence building, resampling, QC
├── labels/        SOC computation (coulomb counting, dynamic capacity, pseudo-labels)
├── models/        BaseSOCModel + VidalLSTM + backbone placeholders
├── transfer/      BaseTransferMethod + source_only/scratch/finetune + placeholders
├── training/      Trainer, loops, checkpointing, callbacks, early stopping
├── evaluation/    Metrics, Evaluator, physical-consistency, domain, plots
└── experiments/   ExperimentRunner, result schema, comparison
```

## Data flow

```
YAML configs
  └─► ExperimentRunner.run()
        ├─► Dataset.generate_cycles() or Dataset.load_cycle()
        ├─► split_cycles()
        ├─► SequenceBuilder.build()
        ├─► MinMaxNormalizer.fit(train) / .transform(all)
        ├─► CycleDataset + DataLoader
        ├─► ModelRegistry.build()
        ├─► TransferMethod.fit_source()  [trains on source]
        ├─► TransferMethod.adapt_target() [optional]
        ├─► Evaluator.evaluate()
        └─► Save: checkpoint / metrics.json / predictions.csv / plots
```
