# Benchmark Experiment Runner

## Quick Start

```bash
python run_experiment.py --config configs/experiments/hex2vec_king_county.yaml
```

## Adding a New Dataset

1. Register it in `src/data/dataset_factory.py`:

```python
DATASET_REGISTRY["MyNewDataset"] = ("srai.datasets", "MyNewDatasetClass")
```

2. If it is multi-city (like AirbnbMulticity), add its key to
   `MULTI_CITY_DATASETS` inside `build_full_regions`.

3. Create an experiment config:

```yaml
# configs/experiments/hex2vec_my_dataset.yaml
experiment_name: "hex2vec_my_dataset_r9"

dataset:
  name: "MyNewDataset"
  version: "9"
  use_numerical_columns: false
  dev_size: 0.1
```

## Adding a New Embedder

1. Register it in `src/embedders/embedder_factory.py`:

```python
EMBEDDER_REGISTRY["MyEmbedder"] = ("srai.embedders", "MyEmbedderClass")
```

2. Add a construction branch in `build_embedder()` if the constructor
   signature differs from the existing embedders.

3. If it does not need a `.fit()` call, add its key to `NO_FIT_EMBEDDERS`.

4. Create an experiment config:

```yaml
embedder:
  name: "MyEmbedder"
  hidden_sizes: [128, 64]
  fit_kwargs:
    batch_size: 256
    trainer_kwargs:
      max_epochs: 5
```

---

## Configuration System

Every experiment config is **merged on top of `configs/base.yaml`**, so you
only need to list the values that differ from the defaults.

Key config sections:

| Section | Purpose |
|---|---|
| `dataset` | Dataset name, version, numerical columns toggle, dev split size |
| `embedder` | Embedder class, architecture, fit kwargs |
| `model` | Regression head layer sizes, dropout |
| `training` | Epochs, batch size, LR, early stopping, loss function |
| `resolution` | H3 resolution (applies to regionaliser and dataset version) |