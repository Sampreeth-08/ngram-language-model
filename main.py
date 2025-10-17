import os
from collections import Counter
from typing import List, Tuple, Sequence, Any
import math
from models import MLEModel, AddOneModel, InterpolationModel, StupidBackoffModel
import random



def load_and_preprocess_data(file_path: str) -> List[List[str]]:
    sentences = []
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                continue
                
            tokens = line.split()
            
            processed_sentence = ['<s>'] + tokens + ['</s>']
            
            sentences.append(processed_sentence)
            
    return sentences

def calculate_perplexity(model: 'MLEModel', test_data: List[List[str]]) -> float:
    """
    Calculates the perplexity of a given model on a test dataset.

    Args:
        model: A trained N-gram model (e.g., an instance of MLEModel).
        test_data: The test dataset, as a list of pre-processed sentences.

    Returns:
        The perplexity (PP) score. Returns float('inf') if a zero
        probability is encountered.
    """
    total_log_prob = 0.0
    test_event_count = 0  # This is 'N' in the formula
    
    n = model.n

    for sentence in test_data:
        if len(sentence) < n:
            continue

        for i in range(len(sentence) - n + 1):
            ngram = tuple(sentence[i : i+n])
            
            prob = model.get_token_probability(ngram)
            
            if prob == 0.0:
                print(f"Zero probability encountered for n-gram: {ngram}")
                print("Perplexity is undefined. Returning 'inf'.")
                return float('inf')
            
            total_log_prob += math.log(prob)
            
            test_event_count += 1

    if test_event_count == 0:
        return 1.0  # Or 0.0, convention for empty test set

    # PP = exp( -(1/N) * total_log_prob )
    avg_log_prob = total_log_prob / test_event_count
    perplexity = math.exp(-avg_log_prob)
    
    return perplexity




def predict_next_word(model: Any, 
                      context: Tuple[str, ...], 
                      vocabulary: List[str]) -> str:
    """
    Predicts the next word by sampling from the model's
    probability distribution given a context.

    Args:
        model: A trained model (e.g., InterpolationModel or StupidBackoffModel).
        context: The preceding n-1 tokens (e.g., ('<s>', '<s>') for n=3).
        vocabulary: A list of all unique words in the vocab.

    Returns:
        The randomly sampled next word.
    """
    words = []
    probabilities = []
    
    n = model.n
    
    if len(context) != n - 1:
        raise ValueError(f"Context length ({len(context)}) must be n-1 ({n-1})")

    for word in vocabulary:
        ngram = context + (word,)
        
        prob = model.get_token_probability(ngram)
        
        words.append(word)
        probabilities.append(prob)

    try:
        chosen_word = random.choices(words, weights=probabilities, k=1)[0]
    except ValueError:
        chosen_word = "<unk>"
        
    return chosen_word

def generate_sentence(model: Any, max_length: int = 50) -> str:
    """
    Generates a new sentence using the provided trained model.

    Args:
        model: A trained model (e.g., InterpolationModel or StupidBackoffModel).
               The model MUST have a 'unigram_model' attribute
               to get the vocabulary.
        max_length: The maximum number of tokens in a sentence
                    before stopping.

    Returns:
        A generated sentence as a string.
    """
    
    try:
        vocab_counts = model.unigram_model.ngram_counts
        vocabulary = [word_tuple[0] for word_tuple in vocab_counts.keys()]
    except AttributeError:
        print("Error: Model must have a 'unigram_model' attribute to get vocabulary.")
        return ""

    n = model.n
    sentence = ["<s>"] * (n - 1)
    
    while len(sentence) < max_length:
        context = tuple(sentence[-(n - 1):])
        
        next_word = predict_next_word(model, context, vocabulary)
        
        sentence.append(next_word)
        
        if next_word == "</s>":
            break
            
    start_index = n - 1
    end_index = -1 if sentence[-1] == "</s>" else len(sentence)
    
    return " ".join(sentence[start_index:end_index])

def main():
    """
    Main function to run all tasks for the N-gram assignment.
    """
    
    print("--- 1. Loading and Pre-processing Data ---")
    data_folder = "ptbdataset"
    train_file = os.path.join(data_folder, "ptb.train.txt")
    valid_file = os.path.join(data_folder, "ptb.valid.txt")
    test_file = os.path.join(data_folder, "ptb.test.txt")
    
    try:
        training_data = load_and_preprocess_data(train_file)
        validation_data = load_and_preprocess_data(valid_file)
        test_data = load_and_preprocess_data(test_file)
        
        if not training_data or not validation_data or not test_data:
            print("\nError: One or more data files are missing or empty.")
            print(f"Please check the folder: '{data_folder}'")
            return
            
        print(f"Loaded {len(training_data)} training sentences.")
        print(f"Loaded {len(validation_data)} validation sentences.")
        print(f"Loaded {len(test_data)} test sentences.")
    
    except Exception as e:
        print(f"\nFailed to load data: {e}")
        print(f"Please ensure your data is in a folder named '{data_folder}'")
        return

    # --- 2. Section 3.1: MLE Model Evaluation ---
    print("\n\n--- 2. Section 3.1: MLE Model Evaluation (N=1,2,3,4) ---")
    for n in [1, 2, 3, 4]:
        model = MLEModel(n=n)
        model.train(training_data)
        pp = calculate_perplexity(model, test_data)
        print(f"Perplexity (Test Data) for n={n} MLE: {pp}")

    # --- 3. Section 3.2.1: Add-1 Smoothing ---
    print("\n\n--- 3. Section 3.2.1: Add-1 Smoothing (n=3) ---")
    add_one_model = AddOneModel(n=3)
    add_one_model.train(training_data)
    pp_add_one = calculate_perplexity(add_one_model, test_data)
    print(f"Perplexity (Test Data) for n=3 Add-1: {pp_add_one}")

    # --- 4. Section 3.2.2: Linear Interpolation ---
    print("\n\n--- 4. Section 3.2.2: Linear Interpolation (Tuning) ---")
    lambda_combinations = [[0.1, 0.3, 0.6], [0.33, 0.33, 0.34], [0.6, 0.3, 0.1], [0.1, 0.8, 0.1]]
    best_lambdas = None
    best_interp_perplexity = float('inf')
    
    # Pre-train the sub-models once for efficiency
    print("Pre-training sub-models for Interpolation/Backoff...")
    shared_unigram = MLEModel(n=1)
    shared_bigram = MLEModel(n=2)
    shared_trigram = MLEModel(n=3)
    shared_unigram.train(training_data)
    shared_bigram.train(training_data)
    shared_trigram.train(training_data)
    print("Sub-models trained.")

    for lambdas in lambda_combinations:
        interp_model = InterpolationModel(lambdas)
        # Assign pre-trained models
        interp_model.unigram_model = shared_unigram
        interp_model.bigram_model = shared_bigram
        interp_model.trigram_model = shared_trigram
        
        pp = calculate_perplexity(interp_model, validation_data)
        print(f"Perplexity (Validation) for Lambdas {lambdas}: {pp}")
        if pp < best_interp_perplexity:
            best_interp_perplexity = pp
            best_lambdas = lambdas
            
    print(f"Optimal Lambdas: {best_lambdas} (PP={best_interp_perplexity})")

    # Final Interpolation Evaluation
    print("--- 4. Final Evaluation (Linear Interpolation) ---")
    final_interp_model = InterpolationModel(best_lambdas)
    final_interp_model.unigram_model = shared_unigram
    final_interp_model.bigram_model = shared_bigram
    final_interp_model.trigram_model = shared_trigram
    pp_interp_final = calculate_perplexity(final_interp_model, test_data)
    print(f"Perplexity (Test Data) for Interpolation: {pp_interp_final}")

    # --- 5. Section 3.2.2: Stupid Backoff ---
    print("\n\n--- 5. Section 3.2.2: Stupid Backoff (Tuning) ---")
    alpha_values = [0.1, 0.4, 1.0]
    best_alpha = None
    best_backoff_perplexity = float('inf')

    for alpha in alpha_values:
        backoff_model = StupidBackoffModel(alpha=alpha)
        # Assign pre-trained models
        backoff_model.unigram_model = shared_unigram
        backoff_model.bigram_model = shared_bigram
        backoff_model.trigram_model = shared_trigram
        
        pp = calculate_perplexity(backoff_model, validation_data)
        print(f"Perplexity (Validation) for Alpha {alpha}: {pp}")
        if pp < best_backoff_perplexity:
            best_backoff_perplexity = pp
            best_alpha = alpha
            
    print(f"Optimal Alpha: {best_alpha} (PP={best_backoff_perplexity})")

    # Final Backoff Evaluation
    print("--- 5. Final Evaluation (Stupid Backoff) ---")
    final_backoff_model = StupidBackoffModel(best_alpha)
    final_backoff_model.unigram_model = shared_unigram
    final_backoff_model.bigram_model = shared_bigram
    final_backoff_model.trigram_model = shared_trigram
    pp_backoff_final = calculate_perplexity(final_backoff_model, test_data)
    print(f"Perplexity (Test Data) for Stupid Backoff: {pp_backoff_final}")

    # --- 6. Section 4.4: Qualitative Analysis (Text Generation) ---
    print("\n\n--- 6. Section 4.4: Text Generation ---")
    
    # Select the best model based on final test perplexity
    if pp_interp_final < pp_backoff_final:
        best_model = final_interp_model
        print(f"Selecting best model: Linear Interpolation (PP={pp_interp_final})")
    else:
        best_model = final_backoff_model
        print(f"Selecting best model: Stupid Backoff (PP={pp_backoff_final})")
        
    print("\nGenerating 5 sentences:")
    for i in range(5):
        generated_text = generate_sentence(best_model, max_length=50)
        print(f"Sentence {i+1}: {generated_text}")
        
    print("\n--- All tasks complete. ---")


if __name__ == "__main__":    
    main()