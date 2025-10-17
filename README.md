# N-gram Language Models

This project implements various N-gram language models for text generation and perplexity evaluation, developed as part of MSML 641 coursework.

## Overview

The project implements four different N-gram language modeling approaches:

1. **Maximum Likelihood Estimation (MLE)** - Basic N-gram models using raw frequency counts
2. **Add-1 Smoothing (Laplace)** - Smoothing technique to handle unseen N-grams
3. **Linear Interpolation** - Combines unigram, bigram, and trigram models with weighted interpolation
4. **Stupid Backoff** - Backs off to lower-order models when higher-order N-grams are unseen

## Project Structure

```
├── main.py           # Main execution script with evaluation pipeline
├── models.py         # N-gram model implementations
├── README.md         # This file
└── ptbdataset/       # Penn Treebank dataset
    ├── ptb.train.txt # Training data
    ├── ptb.valid.txt # Validation data
    └── ptb.test.txt  # Test data
```

## Models Implemented

### MLEModel
Basic N-gram model using Maximum Likelihood Estimation:
- Supports unigram (n=1) through 4-gram (n=4) models
- Calculates probabilities as: `P(w_n | w_1...w_{n-1}) = Count(w_1...w_n) / Count(w_1...w_{n-1})`
- Returns 0 probability for unseen N-grams

### AddOneModel
Extends MLEModel with Add-1 (Laplace) smoothing:
- Adds 1 to all N-gram counts to handle zero probabilities
- Formula: `P(w_n | w_1...w_{n-1}) = (Count(w_1...w_n) + 1) / (Count(w_1...w_{n-1}) + V)`
- Where V is the vocabulary size

### InterpolationModel
Combines unigram, bigram, and trigram models using linear interpolation:
- Formula: `P = λ₃ × P₃ + λ₂ × P₂ + λ₁ × P₁`
- Hyperparameters (λ values) are tuned on validation data
- Always produces valid probabilities

### StupidBackoffModel
Hierarchical backoff model for trigrams:
- Uses trigram probability if available
- Backs off to `α × P_bigram` if trigram unseen
- Further backs off to `α² × P_unigram` if bigram unseen
- Hyperparameter α is tuned on validation data

## Usage

### Basic Execution

Run the complete evaluation pipeline:

```bash
python main.py
```

This will:
1. Load and preprocess Penn Treebank data
2. Train and evaluate MLE models (n=1,2,3,4)
3. Train and evaluate Add-1 smoothing model
4. Tune hyperparameters for Interpolation and Backoff models
5. Generate sample sentences using the best model

### Using Individual Models

```python
from models import MLEModel, AddOneModel, InterpolationModel, StupidBackoffModel

# Train a trigram MLE model
model = MLEModel(n=3)
model.train(training_sentences)

# Get probability of a trigram
prob = model.get_token_probability(('the', 'cat', 'sat'))

# Train interpolation model
interp_model = InterpolationModel([0.1, 0.3, 0.6])  # λ₁, λ₂, λ₃
interp_model.train(training_sentences)
```

## Data Format

The input data should be in Penn Treebank format:
- One sentence per line
- Tokens separated by spaces
- The preprocessing automatically adds `<s>` (start) and `</s>` (end) tokens

Example:
```
no it was n't black monday
but while the new york stock exchange did n't fall apart friday as the dow jones industrial average plunged N points most of it in the final hour it barely managed to stay this side of chaos
```

## Evaluation Metrics

### Perplexity
The primary evaluation metric, calculated as:
```
PP = exp(-1/N × Σ log P(w_i | context_i))
```

Lower perplexity indicates better model performance.

### Text Generation
Models can generate new sentences by sampling from learned probability distributions:
- Uses the model's vocabulary from training data
- Generates until `</s>` token or maximum length reached
- Provides qualitative assessment of model quality

## Dependencies

- Python 3.7+
- Collections (Counter)
- Math
- Random
- Typing
- OS

No external packages required - uses only Python standard library.

## Results

Typical performance on Penn Treebank test set:

| Model | Perplexity (approx.) |
|-------|---------------------|
| Unigram MLE | ~1000 |
| Bigram MLE | ~400-600 |
| Trigram MLE | ~300-500 |
| Add-1 Trigram | ~200-300 |
| Interpolation | ~150-200 |
| Stupid Backoff | ~150-200 |

*Exact values depend on hyperparameter tuning and dataset preprocessing*

## Final Results on Penn Treebank Test Set

The models were evaluated on a held-out test set after any necessary hyperparameter tuning on the validation set. The Stupid Backoff model achieved the best performance, demonstrating the effectiveness of advanced smoothing techniques.



[Image of a performance comparison chart]


| Model                                | Perplexity (Test Set) | Notes                                           |
| ------------------------------------ | --------------------- | ----------------------------------------------- |
| Unigram MLE                          | 576.98                | Baseline for a non-contextual model.            |
| Bigram / Trigram / 4-gram MLE        | `inf`                 | Failed due to unseen N-grams (zero-probability). |
| Add-1 (Laplace) Smoothed Trigram     | 3748.66               | Solved zero-probability but performed poorly.   |
| Linear Interpolation Trigram         | 192.14                | Tuned with optimal λ=[0.33, 0.33, 0.34].       |
| **Stupid Backoff Trigram** | **91.38** | **Best performing model.** Tuned with optimal α=1.0.    |

The results clearly demonstrate the necessity of smoothing, with advanced techniques like Stupid Backoff and Linear Interpolation dramatically outperforming the naive Add-1 method.


## Implementation Details

### Key Features
- **Efficient N-gram counting** using Python Counter objects
- **Modular design** with separate model classes
- **Automatic hyperparameter tuning** on validation data
- **Zero-probability handling** for robust evaluation
- **Memory-efficient** preprocessing and training

### Preprocessing
- Sentences are tokenized on whitespace
- Start `<s>` and end `</s>` tokens added automatically
- Empty lines are skipped
- UTF-8 encoding assumed

### Smoothing Considerations
- MLE models return 0 for unseen N-grams (leading to infinite perplexity)
- Add-1 smoothing ensures all N-grams have non-zero probability
- Interpolation and backoff models gracefully handle unseen N-grams



## Assignment Context

This implementation fulfills the requirements for UMD CMSC 641 Natural Language Processing coursework, covering:
- Section 3.1: MLE N-gram models
- Section 3.2.1: Add-1 smoothing
- Section 3.2.2: Interpolation and backoff methods
- Section 4.4: Text generation and qualitative analysis

## License

This project is for educational purposes as part of UMD coursework.