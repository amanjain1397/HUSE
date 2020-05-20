import pandas as pd
from collections import Counter, OrderedDict
from functools import partial
import re
stopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 
            'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
            'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 
            'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 
            'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 
            'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
            'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 
            'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
            'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', 
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 
            'won', "won't", 'wouldn', "wouldn't"]

def clean_text(x, stopWords):

    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ') # removing the hyphens, slashes and apostrophes; and replace them with a blank space

    for punct in '?!.,"#&$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '') # replace all the common punctuations with no character
    
    x = re.sub('[\d]+', ' ', x) # removes all digit occurences [0-9]
    x = re.sub(r"\b[a-zA-Z]{1,2}\b", "", x) # I also remove all words which are only 1 or 2 characters long, since they are mainly words like 'or', 'an' etc. 
    return ' '.join([word for word in x.split() if not word in stopWords]) # I also remove common stopwords just to be extra sure

def process_text(dataframe):
    # stopWords = list(stopwords.words('english'))
    cleaner = partial(clean_text, stopWords = stopWords) # defining a partial function, so that *clean_text* can be easily used with dataframe 

    dataframe['processed_text'] = dataframe.text.apply(lambda x: "[CLS] " + cleaner(x.lower()) + " [SEP]") # cleaning the text and also inseting the starting and ending tags '[CLS]' ,'[SEP]'
    return dataframe

def process_classes(dataframe):
    dataframe['processed_classes'] = dataframe.classes.str.replace('<', ' ')
    dataframe['processed_classes'] = dataframe['processed_classes'].str.split().apply(lambda x: ' '.join(list(OrderedDict.fromkeys(x))))
    return dataframe

def create_mapped_classes(dataframe):
    classes = sorted(dataframe.processed_classes.unique())
    classes_dict = {v:k for (k,v) in enumerate(classes)}
    dataframe['mapped_classes'] = dataframe.processed_classes.map(classes_dict)
    return dataframe

def process_dataframe(dataframe):
    dataframe = process_text(dataframe)
    dataframe = process_classes(dataframe)
    dataframe = create_mapped_classes(dataframe)
    return dataframe
    
if __name__ == "__main__":
    dataframe = pd.read_csv('original_train_data.csv')
    dataframe = process_dataframe(dataframe)
    print(dataframe.head())
    pass
