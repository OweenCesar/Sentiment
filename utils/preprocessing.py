import re
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import torch


# This file is intented to be used for preprocessing text data for a sentiment analysis task
# [^\w\s]  anything ( that is why [] ) that is not a word character or a whitespace character 


#  tokenizer 
def word_tokenize(text):
    # will return ["This", "is", "a", "test"] if the text is "This is a test"
    return text.split()  # Basic whitespace tokenizer, t

def clean_text(text):
    # the final 'text' will be lowed cased and stripped of leading  spaces. 

    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags ( which start with < and end with >)
    text = re.sub(r'[^\w\s]', ' ', text)  # Detecting  punctuaction ()
    return text.lower().strip()   

def build_vocab(texts, max_words=20000):
    # A vocabulary of the most commons words (20000 words) will be created based on texts. 
    #  Vocab is where the 'magic' happens. enumerate(...) will return an index attached to the most common words (max_words = 20000)
    #  the output of enumerate(..) will be [(0, ("the", 100)), (1, ("cat", 50)), (2, ("dog", 30))]
    #  Since we care only about the first element of the tuple, we will use the _ to ignore the second element.
    #  we assign an index to each word and it will start from 2 since we wanted to save the index 0 and 1 to be used for padding and unknown tokens.

    counter = Counter() # -----> Keep track of the count of each element in the tokens. 
    for text in texts:
        tokens = word_tokenize(text) # Calling the predefined function. 
        counter.update(tokens)   #Increasing the count of each word in the counter 

    vocab = {word: i+2 for i, (word, _) in enumerate(counter.most_common(max_words))} 
    vocab['<pad>'] = 0
    vocab['<unk>'] = 1
    return vocab

def text_to_tensor(texts, vocab, max_len=200):
    # tensors will be initlized as an empty list and then save the tensors here. 
    # In this function, we will store the values from the vocab dictionary into the 'ids' list, ensuring that the length of the list is equal to max_len.
    # If the token was not found then we will use the value assigned to the <unk> token
    # We make sure that the length of the list is equal to max_len by adding <pad> tokens at the end of the list:
    # Dont forget that [vocab['<pad>']] * (max_len - len(ids)) would return 'n' times of 0, in this case 
    
    tensors = []       
    for text in texts:
        tokens = word_tokenize(text)
        ids = [vocab.get(token, vocab['<unk>']) for token in tokens][:max_len]
        ids += [vocab['<pad>']] * (max_len - len(ids))
        tensors.append(torch.tensor(ids, dtype=torch.long))
    return torch.stack(tensors)    


def load_data(filepath):
    # Load the dataset from a CSV file! 
    # Possible fails : Not handling csv files that follow a different format! MUST IMPROVE THIS! 
    # The return format will be of type :  X_train, X_test, y_train, y_test
    df = pd.read_csv(filepath)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0}) # Transforming the categorical values into numerical values
    df['review'] = df['review'].apply(clean_text)
    return train_test_split(df['review'], df['sentiment'], test_size=0.2)  