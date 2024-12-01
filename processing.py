import json
import pandas as pd

def load_data():
    file = open('connections.json')
    connections = json.load(file)
    file.close()

    df = pd.json_normalize(connections, 'answers', ['id', 'date'])
    df_members = pd.DataFrame(df['members'].to_list(), columns=['member1', 'member2', 'member3', 'member4'])
    df = pd.concat([df.drop(columns=['id', 'members']), df_members], axis=1)
    melted_df = df.melt(id_vars=['level', 'date'], value_vars=['member1', 'member2', 'member3', 'member4'],
                        var_name='member_type', value_name='members')
    result_df = melted_df[['level', 'date', 'members']]
    result_df.loc[:, 'members'] = result_df['members'].str.lower()

    return result_df