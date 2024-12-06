from transformers import BertModel, BertTokenizer
import torch
import torch.nn.functional as F
import pandas as pd
from processing import load_data
from sklearn.model_selection import train_test_split
import numpy as np
from testing import accuracy
from helper import contrastive_loss, shuffle
from sklearn.model_selection import KFold
from datetime import datetime

# TO CHANGE:
learning_rate = [1e-5, 1e-4, 1e-2]
small_batch = 10
batch_size = 40
num_epochs = 2
l2_regularization = [1e-2, 1e-3, 1e-5]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.train()

# Get embeddings
def get_embedding(word):
    '''
    Given a word, turns it into a vector embedding
    '''
    inputs = tokenizer(word, return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)  # Average pooling over token embeddings
    return embedding


def bert_fine_tune(df_train_new, optimizer):
    '''
    Fine tunes BERT embeddings based on contrastive loss 
    '''
    # Unfreezing last layer:
    for name, param in model.named_parameters():
        if "encoder.layer.11" not in name:
            param.requires_grad = False

    epoch_loss_list = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        total_epoch_loss = 0

        optimizer.zero_grad() # reset optimizer
        total_loss = torch.tensor(0.0, requires_grad=True)  # reset total_loss for small_batch
        
        for i in range(0, len(df_train_new), small_batch):
            iteration_number = i // small_batch  # Calculate the iteration number
            
            if i % 1000 == 0:
                print(i, iteration_number)
                
            group = df_train_new[i:i+small_batch].reset_index(drop=True)
            n = group.shape[0]

            words_1 = list(group['word_1'])
            words_2 = list(group['word_2'])

            embeds_1 = [get_embedding(x) for x in words_1]
            embeds_2 = [get_embedding(x) for x in words_2]
            
            for j in range(n):
                embed_i1 = embeds_1[j]
                embed_i2 = embeds_2[j]
                label = group.iloc[j]['same_group']

                # Calculate contrastive loss
                loss = contrastive_loss(embed_i1, embed_i2, label)
                total_loss = total_loss + loss # Ensure both are tensors, sum up loss
            

            # Using gradient Accumulation to support larger batch sizes
            if (i + small_batch) % batch_size == 0:
                total_loss.backward() # Backward pass for gradients
                optimizer.step() # update weights
                
                total_epoch_loss += total_loss.item() # Accumulate loss for the epoch

                optimizer.zero_grad() # clear gradients for the next gradient accumulation
                total_loss = torch.tensor(0.0, requires_grad=True)  # Reset total_loss for the next accumulation

        # if any left over loss
        if total_loss.item() > 0:
            total_loss.backward()
            optimizer.step()
            total_epoch_loss += total_loss.item()

        
        epoch_loss_list[epoch] = total_epoch_loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_epoch_loss:.4f}")  # Print total loss for the epoch
    
    print(epoch_loss_list)
    return epoch_loss_list




# Load data and split into train/test for CV
full_df = load_data()

dates = full_df['date'].unique()
from datetime import datetime

# Define the start and end of the range
start_date = datetime.strptime("2023-09-01", "%Y-%m-%d")
end_date = datetime.strptime("2023-09-15", "%Y-%m-%d")

# Filter dates within the range
cv_dates = [x for x in dates if start_date <= datetime.strptime(x, "%Y-%m-%d") <= end_date]

cv_df = full_df[full_df['date'].isin(cv_dates)]



def train_and_validate(cv_df_new, val_data, lr, wd):
    '''
    Performs cross-validation given a train and validation set
    '''

    model = BertModel.from_pretrained('bert-base-uncased')
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    bert_fine_tune(cv_df_new, optimizer)

    val_loss = 0.0
    group = val_data.reset_index(drop=True)
    n = group.shape[0]

    words_1 = list(group['word_1'])
    words_2 = list(group['word_2'])

    embeds_1 = [get_embedding(x) for x in words_1]
    embeds_2 = [get_embedding(x) for x in words_2]

    with torch.no_grad(): # no gradient tracking       
        for j in range(n):
            embed_i1 = embeds_1[j]
            embed_i2 = embeds_2[j]
            label = group.iloc[j]['same_group']

            # Calculate contrastive loss
            loss = contrastive_loss(embed_i1, embed_i2, label)
            if isinstance(loss, torch.Tensor):  
                val_loss += loss.item()  # Extract value from tensor
            else:  
                val_loss += loss  # Add float directly

    return val_loss/n


val_list = {}

# Performs k-fold cross-validation
for lr in learning_rate:
    for wd in l2_regularization:
        val = []
        
        kfold = KFold(n_splits = 5, shuffle = True, random_state=42)

        for fold, (train_indices, val_indices) in enumerate(kfold.split(cv_dates)):
            cv_dates_train = [cv_dates[i] for i in train_indices]
            train_data = full_df[full_df['date'].isin(cv_dates_train)].sort_values(by=['date', 'level']).reset_index(drop=True)

            cv_dates_val = [cv_dates[i] for i in val_indices]
            val_data = full_df[full_df['date'].isin(cv_dates_val)].sort_values(by=['date', 'level']).reset_index(drop=True)

            cv_df_new = shuffle(train_data, cv_dates_train)

            val_df_new = shuffle(val_data, cv_dates_val)
            
            val_fold_loss = train_and_validate(cv_df_new, val_df_new, lr, wd)
            val.append(val_fold_loss)

        avg_val = sum(val) / len(val)
        print(lr, wd, avg_val)
        val_list[(lr, wd)] = avg_val

# Print final results
print("Validation loss results:", val_list)