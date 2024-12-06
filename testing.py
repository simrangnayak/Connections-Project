from itertools import combinations
import numpy as np
import pandas as pd


def calculate_confusion_matrix(predicted_groups, correct_groups):
    '''
    Computes confusion matrix for a single puzzle.
    '''
    all_words = sum(predicted_groups, [])  # Flatten all predicted groups
    TP, FP, TN, FN = 0, 0, 0, 0

    def in_same_group(word1, word2, groups):
        return any(word1 in group and word2 in group for group in groups)

    # Compare all pairs of words
    for word1, word2 in combinations(all_words, 2):
        in_same_pred = in_same_group(word1, word2, predicted_groups)
        in_same_correct = in_same_group(word1, word2, correct_groups)
        
        if in_same_pred and in_same_correct:  # True Positive
            TP += 1
        elif in_same_pred and not in_same_correct:  # False Positive
            FP += 1
        elif not in_same_pred and not in_same_correct:  # True Negative
            TN += 1
        elif not in_same_pred and in_same_correct:  # False Negative
            FN += 1

    # Return confusion matrix in the form of a numpy array
    return np.array([[TP, FN], [FP, TN]])

def compute_metrics_from_cm(cm):
    '''
    Calculates precision, recall, and F1-score from confusion matrix.
    '''
    TP, FN = cm[0]
    FP, TN = cm[1]

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0.0

    return precision, recall, f1_score, specificity



def group_vectors_by_similarity(words, vectors):
    '''
    Clutering words by cosine similarity
    '''
    word_to_vector = {word: vector for word, vector in zip(words, vectors)}

    def find_top_group(word_to_vector):
        '''
        Finding the most likely word grouping
        '''
        group_similarities = []

        if len(word_to_vector) < 4:
            return None, None

        for group_words in combinations(word_to_vector.keys(), 4):
            group_vectors = [word_to_vector[word] for word in group_words]
            similarities = []
            for vec1, vec2 in combinations(group_vectors, 2):
                similarity = np.dot(vec1.T, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                similarities.append(similarity)
            
            avg_similarity = np.mean(similarities)
            group_similarities.append((list(group_words), avg_similarity))

        top_group = max(group_similarities, key=lambda x: x[1])
        return top_group


    groups = []
    remaining_words = words.copy()

    for _ in range(3):
        top_group, avg_similarity = find_top_group({word: word_to_vector[word] for word in remaining_words})
        
        if top_group is None:
            print("Not enough words left to form a complete group.")
            break
        
        groups.append(top_group)
        remaining_words = [word for word in remaining_words if word not in top_group]

    if remaining_words:
        groups.append(remaining_words)

    return groups


def calculate_pairwise_accuracy(predicted_groups, correct_groups, weight_same_group = 2, weight_diff_group = 0.5):
    '''
    Calculating weighted pairwise accuracy for a single puzzle 
    '''
    count = 0
    total_weight = 0
    all_words = sum(predicted_groups, []) 

    def in_same_group(word1, word2, groups):
       return any(word1 in group and word2 in group for group in groups)

    for word1, word2 in combinations(all_words, 2):
        in_same_pred = in_same_group(word1, word2, predicted_groups)
        in_same_correct = in_same_group(word1, word2, correct_groups)
       
        if in_same_pred == in_same_correct == True:
           count += 1*weight_same_group
        if in_same_pred == in_same_correct == False:
            count += 1*weight_diff_group

        # Weighted total pairwise accuracy
        if in_same_correct:
            total_weight += 1*weight_same_group
        else:
            total_weight += 1*weight_diff_group

    accuracy = count / total_weight #instead of 120
    #print(f"Pairwise Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def accuracy(dates_test, df_test, embed_fnc):
    '''
    Calculating pairwise accuracy over all puzzles
    '''
    testing_array = []
    confusion_matrices = []

    for date in dates_test:
        day_df = df_test[df_test['date'] == date]
        day_df_members = day_df['members'] # words for that day

        day_df_embeds = [embed_fnc(x).detach().numpy() for x in day_df_members] # corresponding word embeddings
        embeds_array = np.array(day_df_embeds).squeeze()

        predicted_groups = group_vectors_by_similarity(day_df_members, embeds_array)
        correct_groups = day_df.groupby('level')['members'].apply(list).tolist()

        # Pairwise accuracy
        testing_array.append(calculate_pairwise_accuracy(predicted_groups, correct_groups, weight_same_group=2, weight_diff_group=0.5))

        # Confusion matrix
        cm = calculate_confusion_matrix(predicted_groups, correct_groups)
        confusion_matrices.append(cm)
    
    avg_accuracy = np.mean(testing_array)
    total_cm = np.sum(confusion_matrices, axis=0)
    return avg_accuracy, testing_array, total_cm


