import os
from collections import Counter
from typing import List, Dict, Tuple, Optional, Sequence
import math

def count_ngrams(sentences: List[List[str]], n: int) -> Counter:
    """
    Counts the frequency of all n-grams in a list of pre-processed sentences.

    Args:
        sentences: The list of tokenized sentences (output from load_and_preprocess_data).
        n: The order of the n-gram (e.g., 1 for unigrams, 2 for bigrams).

    Returns:
        A Counter object mapping n-gram tuples to their frequencies.
    """
    counts = Counter()
    
    for sentence in sentences:
        if len(sentence) < n:
            continue
            
        for i in range(len(sentence) - n + 1):
            ngram = tuple(sentence[i : i+n])
            counts[ngram] += 1
            
    return counts

class MLEModel:
    """
    An N-gram Language Model trained using Maximum Likelihood Estimation (MLE).
    """
    
    def __init__(self, n: int):
        """
        Initializes the model.
        
        Args:
            n: The order of the n-gram model (e.g., 1 for unigram, 2 for bigram).
        """
        self.n = n
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        self.total_tokens: int = 0

    def train(self, sentences: List[List[str]]):
        """
        Trains the model on the given sentences.
        
        Args:
            sentences: A list of pre-processed sentences.
        """
        print(f"Training {self.n}-gram model...")
        
        self.ngram_counts = count_ngrams(sentences, self.n)
        
        if self.n == 1:
            self.total_tokens = sum(self.ngram_counts.values())
        else:
            self.context_counts = count_ngrams(sentences, self.n - 1)
            
        print(f"Training complete. Found {len(self.ngram_counts)} unique {self.n}-grams.")

    def get_token_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Calculates the MLE probability of the last word in an n-gram
        given the preceding n-1 words.
        
        Args:
            ngram: The n-gram tuple (e.g., ('am', 'sam', '</s>') for n=3).
        
        Returns:
            The MLE probability, P(w_n | w_1, ..., w_{n-1}).
        """
        if len(ngram) != self.n:
            raise ValueError(f"N-gram tuple length ({len(ngram)}) does not match model n ({self.n})")

        numerator = self.ngram_counts[ngram]
        
        if self.n == 1:
            if self.total_tokens == 0:
                return 0.0 # Avoid division by zero if model is untrained
            return numerator / self.total_tokens

        
        # e.g., for ('am', 'sam', '</s>'), context is ('am', 'sam')
        context = ngram[:-1]
        
        denominator = self.context_counts[context]
        
        if denominator == 0:
            # The context (e.g., 'am sam') was never seen.
            return 0.0
        
        return numerator / denominator
    
class AddOneModel(MLEModel):
    """
    An N-gram Language Model trained using Add-1 (Laplace) Smoothing.
    Inherits from MLEModel.
    """
    
    def __init__(self, n: int):
        """
        Initializes the model.
        
        Args:
            n: The order of the n-gram model (e.g., 1 for unigram, 2 for bigram).
        """
        super().__init__(n)
        
        self.vocab_size: int = 0

    def train(self, sentences: List[List[str]]):
        """
        Trains the model on the given sentences and calculates
        the vocabulary size (V) needed for smoothing.
        
        Args:
            sentences: A list of pre-processed sentences.
        """
        super().train(sentences)
        
        unigram_counts = count_ngrams(sentences, 1)
        self.vocab_size = len(unigram_counts)
        
        print(f"Add-1 Model: Vocabulary size (V) = {self.vocab_size}")

    def get_token_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Calculates the Add-1 smoothed probability of the last word
        in an n-gram.
        
        P_add1 = (Count(ngram) + 1) / (Count(context) + V)
        
        Args:
            ngram: The n-gram tuple (e.g., ('am', 'sam', '</s>') for n=3).
        
        Returns:
            The Add-1 smoothed probability.
        """
        if len(ngram) != self.n:
            raise ValueError(f"N-gram tuple length ({len(ngram)}) does not match model n ({self.n})")

        numerator = self.ngram_counts[ngram] + 1
        
        if self.n == 1:
            denominator = self.total_tokens + self.vocab_size
        else:
            context = ngram[:-1]
            denominator = self.context_counts[context] + self.vocab_size
            
        if denominator == 0:
            return 0.0 
            
        return numerator / denominator
    
class InterpolationModel:
    """
    An N-gram Language Model that uses Linear Interpolation
    to combine unigram, bigram, and trigram probabilities.
    
    This model is hard-coded for n=3 (trigrams) as specified
    in the assignment[cite: 23].
    """
    
    def __init__(self, lambdas: Sequence[float]):
        """
        Initializes the model.
        
        Args:
            lambdas: A sequence (list or tuple) of three floats (lambda1, lambda2, lambda3)
                     representing the weights for the unigram, bigram, and
                     trigram models, respectively.
        """
        if len(lambdas) != 3:
            raise ValueError("Lambdas sequence must contain exactly 3 values.")
        if not math.isclose(sum(lambdas), 1.0):
            print(f"Warning: Lambdas sum to {sum(lambdas)}, not 1.0.")
            
        self.lambda1 = lambdas[0]
        self.lambda2 = lambdas[1]
        self.lambda3 = lambdas[2]
        
        self.n = 3 
        
        self.unigram_model = MLEModel(n=1)
        self.bigram_model = MLEModel(n=2)
        self.trigram_model = MLEModel(n=3)

    def train(self, sentences: List[List[str]]):
        """
        Trains all three internal models (unigram, bigram, trigram)
        on the given sentences.
        
        Args:
            sentences: A list of pre-processed sentences.
        """
        print(f"Training InterpolationModel (Lambdas: {self.lambda1, self.lambda2, self.lambda3})...")
        self.unigram_model.train(sentences)
        self.bigram_model.train(sentences)
        self.trigram_model.train(sentences)
        print("InterpolationModel training complete.")

    def get_token_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Calculates the interpolated probability of the last word in a
        trigram.
        
        P_interp = (lambda3 * P_mle3) + (lambda2 * P_mle2) + (lambda1 * P_mle1)
        
        Args:
            ngram: The n-gram tuple. Must be a trigram (length 3).
        
        Returns:
            The interpolated probability.
        """
        if len(ngram) != self.n:
            raise ValueError(f"N-gram tuple length ({len(ngram)}) does not match model n ({self.n})")

        trigram = ngram
        bigram = ngram[1:]  # e.g., ('am', 'sam', '</s>')
        unigram = ngram[2:] # e.g., ('sam', '</s>')

        p3 = self.trigram_model.get_token_probability(trigram)
        p2 = self.bigram_model.get_token_probability(bigram)
        p1 = self.unigram_model.get_token_probability(unigram)
        
        final_prob = (self.lambda3 * p3) + (self.lambda2 * p2) + (self.lambda1 * p1)
        
        return final_prob
    
class StupidBackoffModel:
    """
    An N-gram Language Model that uses Stupid Backoff.
    
    This model is hard-coded for n=3 (trigrams) to backoff
    to bigrams and unigrams, as specified in the assignment.
    """
    
    def __init__(self, alpha: float = 0.4):
        """
        Initializes the model.
        
        Args:
            alpha: The backoff factor (default is 0.4, as suggested
                   in the assignment).
        """
        self.alpha = alpha
        
        self.n = 3 
        
        self.unigram_model = MLEModel(n=1)
        self.bigram_model = MLEModel(n=2)
        self.trigram_model = MLEModel(n=3)

    def train(self, sentences: List[List[str]]):
        """
        Trains all three internal models (unigram, bigram, trigram)
        on the given sentences.
        
        Args:
            sentences: A list of pre-processed sentences.
        """
        print(f"Training StupidBackoffModel (alpha: {self.alpha})...")
        self.unigram_model.train(sentences)
        self.bigram_model.train(sentences)
        self.trigram_model.train(sentences)
        print("StupidBackoffModel training complete.")

    def get_token_probability(self, ngram: Tuple[str, ...]) -> float:
        """
        Calculates the Stupid Backoff score for a given trigram.
        
        Note: This is not a true probability, but a score used for
        ranking and perplexity calculation.
        
        Args:
            ngram: The n-gram tuple. Must be a trigram (length 3).
        
        Returns:
            The Stupid Backoff score.
        """
        if len(ngram) != self.n:
            raise ValueError(f"N-gram tuple length ({len(ngram)}) does not match model n ({self.n})")

        trigram = ngram
        bigram = ngram[1:]  # e.g., ('am', 'sam', '</s>')
        unigram = ngram[2:] # e.g., ('sam', '</s>')

        p3 = self.trigram_model.get_token_probability(trigram)
        if p3 > 0.0:
            return p3
        
        p2 = self.bigram_model.get_token_probability(bigram)
        if p2 > 0.0:
            return self.alpha * p2
            
        p1 = self.unigram_model.get_token_probability(unigram)
        
        return self.alpha * self.alpha * p1