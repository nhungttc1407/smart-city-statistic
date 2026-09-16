import pandas as pd                                                          
                                                                                
# df = pd.read_csv("Engdata(usage_binatanya_01-03-25_16-11-).csv")              
df = pd.read_csv("merged_conversations.csv")              
                                                                            
print(df.shape)                                                               
print(df.columns.tolist())
print(df.head())
print(len(df))

