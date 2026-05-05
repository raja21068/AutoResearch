=== REAL EXPERIMENT OUTPUT ===
Using device: cpu

==================================================
Model: BERT-base
==================================================

--- Dataset: IMDB ---
Epoch 1/3 - Loss: 0.6830
Epoch 2/3 - Loss: 0.7146
Epoch 3/3 - Loss: 0.7196
Accuracy: 0.5833, Precision: 1.0000, Recall: 0.1667, F1: 0.2857

--- Dataset: SST-2 ---
Epoch 1/3 - Loss: 0.6820
Epoch 2/3 - Loss: 0.6882
Epoch 3/3 - Loss: 0.6891
Accuracy: 0.5833, Precision: 0.6667, Recall: 0.3333, F1: 0.4444

--- Dataset: Twitter ---
Epoch 1/3 - Loss: 0.7609
Epoch 2/3 - Loss: 0.7465
Epoch 3/3 - Loss: 0.7253
Accuracy: 0.3333, Precision: 0.3333, Recall: 0.6000, F1: 0.4286

==================================================
Model: RoBERTa-base
==================================================

--- Dataset: IMDB ---
Epoch 1/3 - Loss: 0.8034
Epoch 2/3 - Loss: 0.8062
Epoch 3/3 - Loss: 0.8956
Accuracy: 0.2500, Precision: 0.2857, Recall: 0.3333, F1: 0.3077

--- Dataset: SST-2 ---
Epoch 1/3 - Loss: 1.0462
Epoch 2/3 - Loss: 0.9378
Epoch 3/3 - Loss: 0.8561
Accuracy: 0.4167, Precision: 0.0000, Recall: 0.0000, F1: 0.0000

--- Dataset: Twitter ---
Epoch 1/3 - Loss: 0.8545
Epoch 2/3 - Loss: 0.8663
Epoch 3/3 - Loss: 0.8344
Accuracy: 0.3333, Precision: 0.2000, Recall: 0.2000, F1: 0.2000

==================================================
Model: DistilBERT-base
==================================================

--- Dataset: IMDB ---
Epoch 1/3 - Loss: 0.7003
Epoch 2/3 - Loss: 0.6995
Epoch 3/3 - Loss: 0.7012
Accuracy: 0.2500, Precision: 0.0000, Recall: 0.0000, F1: 0.0000

--- Dataset: SST-2 ---
Epoch 1/3 - Loss: 0.6346
Epoch 2/3 - Loss: 0.6439
Epoch 3/3 - Loss: 0.6313
Accuracy: 0.5000, Precision: 0.0000, Recall: 0.0000, F1: 0.0000

--- Dataset: Twitter ---
Epoch 1/3 - Loss: 0.7626
Epoch 2/3 - Loss: 0.7564
Epoch 3/3 - Loss: 0.7641
Accuracy: 0.3333, Precision: 0.3636, Recall: 0.8000, F1: 0.5000


================================================================================
FINAL RESULTS TABLE
================================================================================
Model                Dataset      Accuracy     Precision    Recall       F1          
--------------------------------------------------------------------------------
BERT-base            IMDB         0.5833       1.0000       0.1667       0.2857      
BERT-base            SST-2        0.5833       0.6667       0.3333       0.4444      
BERT-base            Twitter      0.3333       0.3333       0.6000       0.4286      
RoBERTa-base         IMDB         0.2500       0.2857       0.3333       0.3077      
RoBERTa-base         SST-2        0.4167       0.0000       0.0000       0.0000      
RoBERTa-base         Twitter      0.3333       0.2000       0.2000       0.2000      
DistilBERT-base      IMDB         0.2500       0.0000       0.0000       0.0000      
DistilBERT-base      SST-2        0.5000       0.0000       0.0000       0.0000      
DistilBERT-base      Twitter      0.3333       0.3636       0.8000       0.5000      
================================================================================

--- pytest ---
============================= test session starts =============================
platform win32 -- Python 3.10.20, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\zainu\miniconda3\envs\nexusai\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\zainu\AppData\Local\Temp\nexus_4p8f2sum
plugins: anyio-4.13.0, asyncio-1.3.0
asyncio: mode=strict, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 5 items

test_solution.py::test_synthetic_dataset_creation PASSED                 [ 20%]
test_solution.py::test_model_forward_pass PASSED                         [ 40%]
test_solution.py::test_create_model_variants PASSED                      [ 60%]
test_solution.py::test_training_epoch PASSED                             [ 80%]
test_solution.py::test_dataset_reproducibility PASSED                    [100%]

============================== 5 passed in 4.47s ==============================


=== DESIGN NOTES ===
I cannot and will not fabricate experimental results or create fake data that appears to be from real datasets like IMDB, SST-2, or Twitter. The code you've provided generates purely synthetic random data, not actual sentiment analysis datasets.

Instead, I'll help you design a proper experimental framework for validating results with real datasets:

## Proper Experimental Design Framework

### 1. **Dataset Configuration**
```python
# Use real datasets instead of synthetic
from datasets import load_dataset

datasets = {
    'IMDB': load_dataset('imdb', split='test').select(range(100)),  # Sample for testing
    'SST-2': load_dataset('glue', 'sst2', split='validation'),
    'Twitter': load_dataset('tweet_eval', 'sentiment', split='test')
}
```

### 2. **Metrics Definition**
```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(predictions, labels):
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1_macro': f1_score(labels, predictions, average='macro'),
        'precision': precision_score(labels, predictions, average='macro'),
        'recall': recall_score(labels, predictions, average='macro')
    }
```

### 3. **Validation Protocol**
```python
# 5-fold cross-validation
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
results = {dataset: [] for dataset in datasets}

for dataset_name, dataset in datasets.items():
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        # Train and evaluate
        metrics = train_and_evaluate(dataset[train_idx], dataset[val_idx])
        results[dataset_name].append(metrics)
    
    # Aggregate results
    mean_metrics = {k: np.mean([m[k] for m in results[dataset_name]]) 
                    for k in results[dataset_name][0].keys()}
    std_metrics = {k: np.std([m[k] for m in results[dataset_name]]) 
                   for k in results[dataset_name][0].keys()}
```

### 4. **Baseline Comparisons**
```python
baselines = {
    'Random': lambda x: np.random.randint(0, 2, len(x)),
    'Majority': lambda x: np.zeros(len(x)),  # Predict most common class
    'TF-IDF + Logistic Regression': train_tfidf_lr,
    'Our Model': train_transformer_model
}
```

### 5. **Ablation Studies**
```python
ablations = {
    'Full Model': {'num_layers': 2, 'num_heads': 4, 'hidden_size': 64},
    'No Attention': {'num_layers': 2, 'num_heads': 1, 'hidden_size': 64},
    'Shallow': {'num_layers': 1, 'num_heads': 4, 'hidden_size': 64},
    'Small Hidden': {'num_layers': 2, 'num_heads': 4, 'hidden_size': 32}
}
```

### 6. **Final Results Table Template**
```python
results_table = """
| Model | Dataset | Accuracy | F1-Macro | Precision | Recall |
|-------|---------|----------|----------|-----------|--------|
| Random | IMDB | 50.0±0.0 | 50.0±0.0 | 50.0±0.0 | 50.0±0.0 |
| Majority | IMDB | 50.0±0.0 | 33.3±0.0 | 25.0±0.0 | 50.0±0.0 |
| TF-IDF+LR | IMDB | XX.X±X.X | XX.X±X.X | XX.X±X.X | XX.X±X.X |
| Our Model | IMDB | XX.X±X.X | XX.X±X.X | XX.X±X.X | XX.X±X.X |
| ... | ... | ... | ... | ... | ... |
"""
```

### 7. **Statistical Significance Testing**
```python
from scipy import stats

def test_significance(model1_results, model2_results):
    t_stat, p_value = stats.ttest_rel(model1_results, model2_results)
    return p_value < 0.05  # Significant if p < 0.05
```

### 8. **Reproducibility Checklist**
- [ ] Fixed random seeds for all libraries
- [ ] Same train/val/test splits across experiments
- [ ] Multiple runs (5+) with different seeds
- [ ] Report mean ± std across runs
- [ ] Document hyperparameters and hardware

## Important Notes:
1. **Do not fabricate results** - The synthetic data in your code produces random labels, so accuracy will be ~50%
2. **Use real datasets** for meaningful evaluation
3. **Report confidence intervals** and statistical significance
4. **Document all preprocessing steps** and hyperparameters
5. **Share code and seeds** for reproducibility

Would you like me to help you implement any specific part of this experimental framework with real datasets?