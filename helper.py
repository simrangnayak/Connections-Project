import torch.nn.functional as F
import pandas as pd
import itertools

# Calculate contrastive loss
def contrastive_loss(embed_1, embed_2, label, m=1.0): 
    '''
    Calculates contrastive loss between two embeddings
    '''
    cosine_similarity = F.cosine_similarity(embed_1, embed_2)
    distance = 1 - cosine_similarity
    if label == 1:  # not in same group
        return 0.5 * max(0, m - distance)**2
    else:  # in the same group
        return 0.5 * distance**2

# Shuffle training data
def shuffle(full_data, dates):
    '''
    Turns each puzzle grouping into 120 word pairings with a flag for group label
    '''
    df_shuffled = pd.DataFrame(columns=['word_1', 'word_2', 'same_group'])

    for date in dates:
        data = full_data[full_data['date'] == date].reset_index(drop=True)
        members = data['members']
        pairs = [list(subset) for subset in itertools.combinations(members, 2)]
        df = pd.DataFrame(pairs, columns=["word_1", "word_2"])
        
        n = df.shape[0]
        for i in range(n):
            word_1 = df['word_1'].iloc[i]
            idx_word_1 = (data.index[data['members'] == word_1].to_list())[0]
            
            level_word_1 = data['level'].iloc[idx_word_1]


            word_2 = df['word_2'].iloc[i]
            idx_word_2 = (data.index[data['members'] == word_2].to_list())[0]
            level_word_2 = data['level'].iloc[idx_word_2]

            if level_word_1 == level_word_2:
                df.loc[i,'same_group'] = 0 # in the same group
            else:
                df.loc[i,'same_group'] = 1 # not in the same group
        
        if df_shuffled.empty:
            df_shuffled = df.copy()
        else:
            df_shuffled = pd.concat([df_shuffled, df], ignore_index=True)
    
    df_shuffled['same_group'] = df_shuffled['same_group'].astype(int)
    df_shuffled = df_shuffled.sample(frac=1).reset_index(drop=True)
    
    return df_shuffled