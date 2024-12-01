from itertools import combinations
import numpy as np
import pandas as pd

def group_vectors_by_similarity(words, vectors):
    
    word_to_vector = {word: vector for word, vector in zip(words, vectors)}

    def find_top_group(word_to_vector):
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
    testing_array = []

    for date in dates_test:
        day_df = df_test[df_test['date'] == date]
        day_df_members = day_df['members'] # words for that day

        day_df_embeds = [embed_fnc(x).detach().numpy() for x in day_df_members] # corresponding word embeddings
        embeds_array = np.array(day_df_embeds).squeeze()

        predicted_groups = group_vectors_by_similarity(day_df_members, embeds_array)
        correct_groups = day_df.groupby('level')['members'].apply(list).tolist()

        testing_array.append(calculate_pairwise_accuracy(predicted_groups, correct_groups, weight_same_group = 2, weight_diff_group = 0.5))
        #print(f"Pairwise Accuracy: {testing_array[i] * 100:.2f}%")
        #print("Iteration " + str(i))
    
    print(testing_array)
    print(np.mean(testing_array))

    return testing_array