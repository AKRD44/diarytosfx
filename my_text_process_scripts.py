import pickle
import pprint
import tempfile
import logging
import numpy as np
import pandas as pd
import os
import re
from gensim.models.phrases import Phrases, Phraser
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora
import gensim
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.collocations import TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()


pp = pprint.PrettyPrinter(indent=4)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_obj(object, name):

    filehandler = open(name+'.pkl', 'w')
    pickle.dump(object, filehandler)


def penn2morphy(penntag, returnNone=False):
    morphy_tag = {'NN': wn.NOUN, 'JJ': wn.ADJ,
                  'VB': wn.VERB, 'RB': wn.ADV}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return None if returnNone else ''


def my_lemmatize(text_processed):  # expects list of strings
    lemmatized_words = []
    for word, tag in pos_tag(text_processed):
        wntag = penn2morphy(tag)
        if wntag:
            lemmatized_words.append(wnl.lemmatize(word, pos=wntag))
        else:
            lemmatized_words.append(word)
    return lemmatized_words


def tokenize_process(text, bigram=False):

    # print("BEFORE")
    # print(text)
    pattern = re.compile('<[^>]*>')  # take out html
    text = re.sub(pattern, '', text)
    # print("AFTER")
    # print(text)

    text = text.replace("/", " ").replace("\n", " ").replace("\r",
                                                             " ").replace("\t", " ").replace("--", " ").replace("_", " ").rstrip()

    # Replace breaks with spaces

    #text = text.replace('<br />', ' ')

    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        text = text.replace(char, ' ' + char + ' ')

    pattern = re.compile(r'\D*')  # matches any NONdigits
    result = " ".join(pattern.findall(text))
    pattern = re.compile(r'\w*')  # takes out non alpha numeric characters
    result = " ".join(pattern.findall(result))
    text = result.replace("_", " ").replace("extra", "")

    # tokenizing
    # RegexpTokenizer is to tokenize according to a regex pattern instead of just " "
    tokenizer = RegexpTokenizer(r'\w+')
    text_processed = tokenizer.tokenize(text)

    # removing any stopwords														stopwords.words('english')
    text_processed = [word.lower() for word in text_processed if word.lower(
    ) not in STOPWORDS and len(word) > 1]

    if bigram:
        print("got to bigram part")

        text_processed = bigram[text_processed]

    return text_processed


class text_tokenize_gen(object):
    def __init__(self, texts):
        self.texts = texts

    def __iter__(self):

        for text in self.texts:

            tokenized_text = tokenize_process(text)

            empty = []
            if tokenized_text == empty:
                tokenized_text = [" "]

            yield tokenized_text

# IMPLEMENT LIKE THIS
#bigram = Phraser(Phrases(text_tokenize_gen(texts)))
#text_processed=[text for text in my_lemmatize(bigram[text_tokenize_gen(texts)])]


def load_data(data_folder, use_old_models):

    try:
        1/use_old_models  # if use_old_models=0, then this fails

        clean_data = pd.read_csv(os.path.join(
            data_folder, "clean_data.csv"), encoding="ISO-8859-1")

        nlp_dict = corpora.Dictionary.load(
            os.path.join(data_folder, 'nlp_dict.dict'))
        processed_texts = np.load(os.path.join(
            data_folder, "processed_texts.npy")).tolist()

        print("loaded preprocessed df")

        bigram = Phraser.load(os.path.join(data_folder, 'bigram'))

    except:
        print("new preprocessing")

        cols_to_use = ['age', 'body_type', 'diet', 'drinks', 'drugs', 'education', 'essay0',
                       'essay1', 'essay2', 'essay5', 'essay6', 'essay7',
                       'ethnicity', 'income', 'job',  'orientation', 'pets',
                       'religion', 'smokes', 'speaks', 'status']

        data = pd.read_csv(os.path.join(
            data_folder, "profiles.csv"), usecols=cols_to_use)

        data.columns = ['age', 'text_body_type', 'text_diet', 'text_drinks', 'text_drugs', 'text_education', 'text_self_sum',
                        'text_life', 'text_goodat', 'text_6things', 'text_thinking', 'text_friday',
                        'text_ethnicity', 'income', 'text_job',  'text_orientation', 'text_pets',
                        'text_religion', 'text_smokes', 'text_speaks', 'text_status']

        # cleaning diet
        data["text_diet"] = data.text_diet.str.replace("strictly ", "").str.replace(
            "mostly ", "").str.replace("other", "anything")
        data["text_diet"] = data.text_diet.replace(np.nan, "anything")

        # cleaning body type
        data["text_body_type"] = data.text_diet.replace(np.nan, "average")

        # cleaning drinks
        data["text_drinks"] = data.text_drinks.replace(np.nan, "socially")

        # cleaning drugs
        data["text_drugs"] = data.text_drugs.replace(np.nan, "never")

        # cleaning education

        data["text_education"] = data.text_education.replace(
            np.nan, "high school")

        data.loc[data.text_education.str.contains(
            "space"), "text_education"] = "high school"

        searchfor = ['university', 'college']
        data.loc[data.text_education.str.contains(
            '|'.join(searchfor)), "text_education"] = 'bachelor'

        searchfor = ['masters', 'law', 'med']
        data.loc[data.text_education.str.contains(
            '|'.join(searchfor)), "text_education"] = 'masters'

        data.loc[data.text_education.str.contains(
            'ph.d'), "text_education"] = 'ph.d'
        data.loc[data.text_education.str.contains(
            'high school'), "text_education"] = 'high school'

        clean_data = data

        columns_with_text = [each_text_col for each_text_col in clean_data.columns.tolist(
        ) if "text" in each_text_col]

        for each_text_col in columns_with_text:
            clean_data[each_text_col] = clean_data[each_text_col].replace(
                np.nan, "")
            clean_data[each_text_col].apply(str)

        clean_data['all_texts'] = clean_data[columns_with_text].apply(
            lambda x: ' / '.join(x), axis=1)

        clean_data = clean_data[(clean_data["all_texts"].str.len() < 21000) & (
            clean_data["all_texts"].str.len() > 860)]  # want text of a minimum size

        clean_data.to_csv(os.path.join(data_folder, "clean_data.csv"))

        # train the bigram
        bigram = Phraser(Phrases(text_tokenize_gen(
            clean_data.all_texts.values.tolist())))

        processed_texts = [[text for text in my_lemmatize(
            texts)] for texts in bigram[text_tokenize_gen(clean_data.all_texts.values)]]

        np.save(os.path.join(data_folder, "processed_texts"), processed_texts)

        bigram.save(os.path.join(data_folder, 'bigram'))
        nlp_dict = corpora.Dictionary(processed_texts)
        # in case you want to filter out some words
        nlp_dict.filter_extremes(no_below=0.1, no_above=0.4)
        # store the dictionary, for future reference
        nlp_dict.save(os.path.join(data_folder, 'nlp_dict.dict'))
        nlp_dict = nlp_dict.load(os.path.join(data_folder, 'nlp_dict.dict'))

        bigram["high school".split()]  # at least I know it works

    return nlp_dict, bigram, clean_data, processed_texts


##### old stuff###
"""
#the problem with this is that the bigram needs a sentence stream in order to learn what words go together. In this case we don't give it a stream, just 1 sentence. 
#it never learns	
def text_process(text):

	text=text.replace("/"," ").replace("\n"," ").replace("\r"," ").replace("\t"," ").replace("--"," ").replace("_"," ").rstrip()

		# Replace breaks with spaces
	text = text.replace('<br />', ' ')

	# Pad punctuation with spaces on both sides
	for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
		text = text.replace(char, ' ' + char + ' ')

	pattern=re.compile(r'\D*')	#matches any NONdigits
	result=" ".join(pattern.findall(text))
	pattern=re.compile(r'\w*')	 #takes out non alpha numeric characters
	result=" ".join(pattern.findall(result))
	text=result.replace("_"," ").replace("extra","")
		
	# tokenizing
	tokenizer = RegexpTokenizer(r'\w+') #RegexpTokenizer is to tokenize according to a regex pattern instead of just " "
	text_processed=tokenizer.tokenize(text)

	# removing any stopwords														stopwords.words('english')
	text_processed = [word.lower() for word in text_processed if word.lower() not in STOPWORDS and len(word)>1]

	#BIGRAM #messes up over here
	bigram = Phrases(text_processed, min_count=20) 
	
	bigram_phraser = gensim.models.phrases.Phraser(bigram)
	text_processed=bigram_phraser[text_processed]#this takes in ['word1','word2','etc']
	#text_processed = [bigram[text] for text in text_processed]

	#BEFORE I USED TO DO THIS FOR BIGRAMS, BUT APPARENTLY PHRASER IS BETTER
	#bigram = Phrases(text_processed, min_count=20) 
	#text_processed=bigram[text_processed]#this takes in ['word1','word2','etc']
	#text_processed = [bigram[text] for text in text_processed]
	

	# Lemmatizing
	#need to lemmatize according to each words tag. So if you're dealing with a verb, lemmatize to get the "base" verb
	#ex: running --> run   if you don't specify a tag, it assumes everything is a noun
	# n for noun files, v for verb files, a for adjective files, r for adverb files.
	
	#print(text_processed)
	text_processed = my_lemmatize(text_processed)#expects list of strings

	return " ".join(text_processed)
	
	

class text_process_gen(object):   #OLD
	def __init__(self, texts):
		self.texts = texts		
	def __iter__(self):
		print(self.texts.shape)
		
		for text in self.texts:
			
			tokenized_text=text_process(text)
			empty=[]
			if tokenized_text == empty:
				tokenized_text=[" "]
			yield tokenized_text
		

def my_tokenizer(text):
	text=text.replace("/"," ").replace("\n"," ").replace("\r"," ").replace("\t"," ").replace("--"," ").replace("_"," ").rstrip()
	
		# Replace breaks with spaces
	text = text.replace('<br />', ' ')

	# Pad punctuation with spaces on both sides
	for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
		text = text.replace(char, ' ' + char + ' ')
		
	pattern=re.compile(r'\D*')	#matches any NONdigits
	result=" ".join(pattern.findall(text))
	pattern=re.compile(r'\w*')	 #takes out non alpha numeric characters
	result=" ".join(pattern.findall(result))
	text=result.replace("_"," ").replace("extra","")
	#pattern=re.compile('<.*?>')   #take out html
	pattern=re.compile('<[^>]*>')   #take out html
	text = re.sub(pattern, '', text)

	
	# tokenizing
	tokenizer = RegexpTokenizer(r'\w+') #RegexpTokenizer is to tokenize according to a regex pattern instead of just " "
	text_processed=tokenizer.tokenize(text)


	# removing any stopwords														stopwords.words('english')
	text_processed = [word.lower() for word in text_processed if word.lower() not in STOPWORDS and len(word)>1]

	return text_processed
		
"""
