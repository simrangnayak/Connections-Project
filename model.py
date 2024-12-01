from transformers import BertModel, BertTokenizer
import torch
import torch.nn.functional as F
import pandas as pd
from processing import load_data
from sklearn.model_selection import train_test_split
import numpy as np
from testing import accuracy
from helper import contrastive_loss, shuffle

# TO CHANGE:
learning_rate = 1e-4 #1e-4
small_batch = 20
batch_size = 120
num_epochs = 8
test_train_split = 0.7
l2_regularization = 1e-5 #1e-5
model = 'BERT' # Can choose between {'BERT', 'ALBERT', 'ROBERTA'}

# BERT model
if model == 'BERT':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

# ALBERT model
if model == 'ALBERT':
    from transformers import AlbertModel, AlbertTokenizer
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    model = AlbertModel.from_pretrained('albert-base-v2')

# ROBERTA model
if model == 'ROBERTA':
    from transformers import RobertaModel, RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')

# Train and initialize model
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = l2_regularization)

# Get embeddings
def get_embedding(word):
    inputs = tokenizer(word, return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)  # Average pooling over token embeddings
    return embedding


# Load data and split into train/test
full_df = load_data()

dates = full_df['date'].unique()
dates_train, dates_test = train_test_split(dates, train_size = test_train_split)

df_train = full_df[full_df['date'].isin(dates_train)].sort_values(by=['date', 'level']).reset_index(drop=True)
df_test = full_df[full_df['date'].isin(dates_test)].sort_values(by=['date', 'level']).reset_index(drop=True)

# Shuffle training data
df_train_new = shuffle(df_train, dates_train)
print(len(df_train_new))

def bert_fine_tune():

    # Unfreezing last layer:
    #for name, param in model.named_parameters():
    #    if "encoder.layer.11" not in name:
    #        param.requires_grad = False

    # Unfreezing last 2 layers:
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "encoder.layer.10" in name or "encoder.layer.11" in name:
            param.requires_grad = True

    epoch_loss_list = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        total_epoch_loss = 0

        optimizer.zero_grad() # reset optimizer
        total_loss = torch.tensor(0.0, requires_grad=True)  # reset total_loss for small_batch
        
        
        for i in range(0, len(df_train_new), small_batch):
            iteration_number = i // small_batch  # Calculate the iteration number
            
            if i % 1000 == 0:
                print(i, iteration_number)
            #print(i)
            #print(f"Iteration number: {iteration_number}")  # Debugging line to track iteration number
                
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
                # Backward pass for gradients
                total_loss.backward()
                optimizer.step() # update weights
                
                total_epoch_loss += total_loss.item() # Accumulate loss for the epoch

                #print("Computed gradients and backward pass")

                optimizer.zero_grad() # clear gradients for the next gradient accumulation
                total_loss = torch.tensor(0.0, requires_grad=True)  # Reset total_loss for the next accumulation

        # if any left over loss
        if total_loss.item() > 0:
            total_loss.backward()
            optimizer.step()
            total_epoch_loss += total_loss.item()

        
        epoch_loss_list[epoch] = total_epoch_loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_epoch_loss:.4f}")  # Print total loss for the epoch
    
    # Print training accuracy here:
    accuracy(dates_train, df_train, get_embedding)

    print(epoch_loss_list)
    return epoch_loss_list

# Pre-bert model
accuracy(dates_test, df_test, get_embedding)

# Testing
bert_fine_tune()

# Post-bert model
accuracy(dates_test, df_test, get_embedding)