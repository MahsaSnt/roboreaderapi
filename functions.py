from wordcloud import WordCloud, STOPWORDS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
import collections
from nltk import ngrams
from nltk import word_tokenize
import heapq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import re
from newspaper import Article
import nltk
from nltk.corpus import stopwords
from nltk.downloader import download, download_shell 

#download package from nltk
# download('punkt',quiet=True)
# download('wordnet',quiet=True)
# download('stopwords',quiet=True)


def clean_text(url):
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()
    corpus = article.text

    #tokenization
    sent_tokens = nltk.sent_tokenize(corpus)
    text = ''
    sentences = []
    for s in sent_tokens:
        s = re.sub(r'\d+', ' ', s)
        s = re.sub(r'\[[0-9]*\]', ' ', s)
        s = re.sub(r'\s+', ' ', s)
        # Removing special characters and digits
        s = re.sub('[^a-zA-Z]', ' ', s)
        s = re.sub(r'\s+', ' ', s)
        if s[0] == ' ':
            s = s[1:]
        if s[-1] == ' ':
            s = s[:-1]
        sentences.append(s)
        text += s 
    
    lang_stopwords = stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    words = [w.lower() for w in tokens if w.lower() not in string.punctuation and w.lower() not in lang_stopwords]
    
    text = ' '.join(words)
    
    return [text, sentences, words]

    
def word_cloud(n,text):
    img = BytesIO()
    wordcloud = WordCloud(width = 1000, height = 600, background_color = 'white', collocations=False, stopwords = set(STOPWORDS), min_font_size = int(n)).generate(text)
    plt.figure(figsize = (8, 4), facecolor = None) 
    plt.axis("off") 
    plt.tight_layout(pad = 0)
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url
    

def n_gram(n_words, n_ngrams, text):
    ngram_counts = collections.Counter(ngrams(text.split(), int(n_words)))
    nn = ngram_counts.most_common(int(n_ngrams))

    a = np.array(nn)
    objects = a[:,0]
    y_pos = np.arange(len(objects))
    performance = a[:,1]
    
    objects = objects[::-1]
    performance = performance[::-1]
    
    img = BytesIO()
    plt.barh(y_pos, performance, align='center', alpha=0.5)
    plt.yticks(y_pos, objects)
    
    plt.xlabel('Counts')
    plt.title('Top '+ n_ngrams +' '+ n_words +'-grams Frequency Plot')
    
    plt.tight_layout(pad = 0)
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url
    


def dispersion_plot(text, words):
    words_token = word_tokenize(text)
    points = [(x,y) for x in range(len(words_token)) for y in range(len(words)) if words_token[x] == words[y]]

    if points:
        x,y=zip(*points)
    else:
        x=y=()

    img = BytesIO()
    plt.plot(x,y,"r|",scalex=0.1)
    plt.yticks(range(len(words)),words,color="b")
    plt.ylim(-1,len(words))
    plt.title("Lexical Dispersion Plot")
    plt.xlabel("Word Offset")

    plt.tight_layout(pad = 0)
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url
    
def summary(n_sentences, sentences, words):
    word_frequencies = {}
    for word in words:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1
    
    maximum_frequncy = max(word_frequencies.values())
    
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
    
    sentence_scores = {}
    for sent in sentences:
        for word in words:
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(int(n_sentences), sentence_scores, key=sentence_scores.get)
    summary_text = '.'.join(summary_sentences) + '.'
    return summary_text.replace(' .','.')


remove_punct_dict = dict((ord(punct),None) for punct in string.punctuation)
def LemNormalize(text1):
    return word_tokenize(text1.lower().translate(remove_punct_dict))

def response(user_response, sentences):
  user_response = user_response.lower()
  if user_response in ["hi","hello","hey","hola"]:
      robo_response = 'hello, I like to help you, please ask your questions.'
      
  elif user_response in ['bye', 'goodbye']:
      robo_response = 'See you later'
      
  elif user_response in ['thanks', 'thank you', 'good', 'nice']:
      robo_response = 'your welcome'        
  
  else:
      sentences.append(user_response)
      tfidfvec=TfidfVectorizer(tokenizer = LemNormalize , stop_words='english')
      tfidf=tfidfvec.fit_transform(sentences)
      
      val = cosine_similarity(tfidf[-1],tfidf[:-1])
    
      idx = val.argsort()[0][-2]
      flat = val.flatten()
      flat.sort()
      score = flat[-2]
    
      if score==0:
        robo_response = "sorry,I dont understand"
      else:
        robo_response = sentences[idx]
        
      sentences.remove(user_response)
  return robo_response