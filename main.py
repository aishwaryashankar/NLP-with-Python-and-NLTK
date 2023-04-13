import os
import nltk
import nltk.corpus

print(os.listdir(nltk.data.find("corpora")))

from nltk.corpus import brown

print(brown.words())

print(nltk.corpus.gutenberg.fileids())
hamlet = nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')
print(hamlet)
print("--------------------")

for word in hamlet[:500]:
  print(word, sep=' ', end=' ')
print()
print("-------------------------")
'''
The below is a paragraph taken from Investopedia's article - Artificial Intelligence: What It Is and How It Is Used.

Using the paragraph to explore functionalities of NLTK
'''

AI_para = '''
Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.

The ideal characteristic of artificial intelligence is its ability to rationalize and take actions that have the best chance of achieving a specific goal. A subset of artificial intelligence is machine learning (ML), which refers to the concept that computer programs can automatically learn from and adapt to new data without being assisted by humans. Deep learning techniques enable this automatic learning through the absorption of huge amounts of unstructured data such as text, images, or video.
'''
print(type(AI_para))

from nltk.tokenize import word_tokenize

AI_para_tokens = word_tokenize(AI_para)
print(AI_para_tokens)

print(len(AI_para_tokens))

from nltk.probability import FreqDist

fdist = FreqDist()
for word in AI_para_tokens:
  fdist[word.lower()] += 1
print(fdist)
print(fdist['artificial'])
print(len(fdist))

fdist_top10 = fdist.most_common(10)
print(fdist_top10)

# to tokenize by paragraphs (separated by blank line)
from nltk.tokenize import blankline_tokenize

AI_blank = blankline_tokenize(AI_para)
print(len(AI_blank))
print(AI_blank[0])
print(AI_blank[1])

from nltk.util import bigrams, trigrams, ngrams

string = "The best and most beautiful things in the world cannot be seen or even touched, they must be felt with the heart"
quotes_tokens = word_tokenize(string)
print(quotes_tokens)

quotes_bigrams = list(bigrams(quotes_tokens))
print(quotes_bigrams)

quotes_trigrams = list(trigrams(quotes_tokens))
print(quotes_trigrams)

quotes_ngrams = list(ngrams(quotes_tokens, 5))
print(quotes_ngrams)

print("PorterStemmer Used:")
from nltk.stem import PorterStemmer

pst = PorterStemmer()
pst.stem("having")
words_to_stem = ["give", "giving", "given", "gave"]
for word in words_to_stem:
  print(word + " - Stem: " + pst.stem(word))

print("Lancaster Stemmer Used:")
from nltk.stem import LancasterStemmer

lst = LancasterStemmer()
lst.stem("having")
for word in words_to_stem:
  print(word + " - Stem: " + lst.stem(word))

print("Snowball Stemmer Used: ")
from nltk.stem import SnowballStemmer

sbst = SnowballStemmer('english')
for word in words_to_stem:
  print(word + " - Stem: " + sbst.stem(word))

from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer

word_lem = WordNetLemmatizer()
print(word_lem.lemmatize("corpora"))

from nltk.corpus import stopwords

print(stopwords.words('english'))
print(len(stopwords.words('english')))

import re

punctuation = re.compile(r'[-.?!,:;()|0-9]')
post_punctuation = []
for words in AI_para_tokens:
  word = punctuation.sub("", words)
  if len(word) > 0:
    post_punctuation.append(word)
print(post_punctuation)

# determines part of speech for each token
sent = "Timothy is a natural when it comes to drawing"
sent_tokens = word_tokenize(sent)
for token in sent_tokens:
  print(nltk.pos_tag([token]))

from nltk import ne_chunk

NE_sent = "The US President stays in the WHITE HOUSE"
NE_tokens = word_tokenize(NE_sent)
NE_tags = nltk.pos_tag(NE_tokens)
NE_NER = ne_chunk(NE_tags)
print(NE_NER)

new = "The big cat ate the little mouse who was after fresh cheese"
new_tokens = nltk.pos_tag(word_tokenize(new))
print(new_tokens)
grammar_np = r"NP: {<DT>?<JJ>*<NN>"}"
chunk_parser = nltk.RegexParser(grammar_np)
chunk_result = chunk_parser.parse(new_tokens)
print(chunk_result)
