from wordcloud import WordCloud, STOPWORDS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO,StringIO
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
from yellowbrick.text import PosTagVisualizer
from nltk.tag import pos_tag
from collections import Counter
# from gensim.summarization import keywords
from textblob import Word
import PyPDF2  
import docx2txt
import librosa
import soundfile as sf
import wave
import speech_recognition as sr 
import os 
from pydub import AudioSegment
from pydub.silence import split_on_silence
import random
import shutil
#import spacy

# Load English tokenizer, tagger, parser and NER
# python -m spacy download en_core_web_sm
#nlp = spacy.load("en_core_web_sm")


#download package from nltk
# download('punkt',quiet=True)
# download('wordnet',quiet=True)
# download('stopwords',quiet=True)
# download('averaged_perceptron_tagger')

def xcut(filename, dr,  min_silence_len = 500):
    n = random.randint(0,1000)
    directory = dr + '/chunk' + str(n)
    if os.path.exists(directory):
            shutil.rmtree(directory)
    os.makedirs(directory)    
    newname = 'tmp' + str(n) +'.wav'
    x,_ = librosa.load(filename, sr=16000)
    sf.write(directory + '/' + newname, x, 16000)
    text = ''
    cnt = 0
    sound = AudioSegment.from_wav(directory + '/' + newname)
    chunks = split_on_silence(sound, min_silence_len=min_silence_len, silence_thresh=-36, keep_silence=400)#silence time:700ms and silence_dBFS<-70dBFS
    
    for ck in chunks:
        path = directory + '/tmp' + ('_%d.wav'%cnt)
        nck = ck.set_frame_rate(16000)
        nck = nck.set_channels(1)
        nck.export(path, format="wav")
        r = sr.Recognizer()
        try:
            with sr.AudioFile(path) as source:
                audio_data = r.record(source)
                tx = r.recognize_google(audio_data)
                text += tx + '. '
                cnt = cnt + 1
        except:
            pass
    shutil.rmtree(directory)
    
    return text[:-1]

def give_text(typ, path, dr, min_silence_len = 500):
    if typ == 'url':
        article = Article(path)
        article.download()
        article.parse()
        article.nlp()
        text = article.text  
    elif typ == 'text_file':
        ext = path.rsplit(".", 1)[1].lower()
        if ext == 'docx':
            text = docx2txt.process(path)
        elif ext == 'pdf':                    
            pdfFileObj = open(path, 'rb')   
            pdfReader = PyPDF2.PdfFileReader(pdfFileObj)  
            n_page=pdfReader.numPages 
            text = ''
            for n in range(n_page):
                pageObj = pdfReader.getPage(n)
                text += pageObj.extractText()
        elif ext == 'txt':                    
            article = open(path,'r')
            text = ''
            for line in article:
                text += str(line)  
        else:
            text = 'please input a valid file/url' 
    elif typ == 'audio_file' :
        text = xcut(path, dr, min_silence_len)
    else :
        text = 'please input a valid file/url'
    return text    

def clean_text(corpus):
    #tokenization
    sent_tokens = nltk.sent_tokenize(corpus)
    text = ''
    #postag = []
    lem_text = ''
    lang_stopwords = stopwords.words('english')
    for s in sent_tokens:
        #s = re.sub(r'\d+', ' ', s)
        s = re.sub('\n\n\n', '. ', s)
        s = re.sub('\n\n', '. ', s)
        s = re.sub('\n', '. ', s)
        s = re.sub(r'\[[0-9]*\]', ' ', s)
        s = re.sub(r'\s+', ' ', s)
        s = s.replace('\"', '')
        s = s.replace('\\', '')
        s = re.sub(r'^[.*]$', '',s)
        s = re.sub(r'\[[a-z]*\]', '',s)
        s = re.sub(r'\[[ a-z ]*\]', '',s)
        s = s.replace(' .', '.')
        # Removing special characters and digits
        #s = re.sub('[^a-zA-Z]', ' ', s)
        if s[0] == ' ':
            s = s[1:]
        if s[-1] == ' ':
            s = s[:-1]
        text += s + ' '    
        s1 = re.sub('[^a-zA-Z]', ' ', s)
        #s1 =  " ".join([token.lemma_ for token in nlp(s1)])
        
        tokens = nltk.word_tokenize(s1)
        word = [w.lower() for w in tokens if w.lower() not in string.punctuation]

        k = [pos_tag(word)][0]
        for i in k:
            if i[0] not in lang_stopwords:  
                w = Word(i[0])
                if i[1][0] == 'V':
                    l = w.lemmatize('v')
                elif i[1][0] == 'N':
                    l = w.lemmatize('n')
                elif i[1][0] == 'J':
                    l = w.lemmatize('a')  
                elif i[1][0] == 'R':
                    l = w.lemmatize('r') 
                else :
                    l = w.lemmatize()
                lem_text += l + ' '
        
        # tokens = nltk.word_tokenize(s1)
        # word = [w.lower() for w in tokens if w.lower() not in string.punctuation and w.lower() not in lang_stopwords]
        # lem_text += ' '.join(word)
   
    
      
    return [text[:-1],lem_text[:-1]]


def word_cloud(n,text):
    img = BytesIO()
    wordcloud = WordCloud(width = 1000, height = 600, background_color = 'white', collocations=False, stopwords = set(STOPWORDS), min_font_size = int(n)).generate(text)
    plt.figure(figsize = (8, 4), facecolor = None) 
    plt.axis("off") 
    plt.imshow(wordcloud)
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
    words = words.split(",")
    for i,word in enumerate(words):
        if word[0]==" ":
            word = word[1:]
        if word[-1]==" ":
            word = word[:-1]
        words[i] = word 
        
    words_token = word_tokenize(text)
    points = [(x,y) for x in range(len(words_token)) for y in range(len(words)) if words_token[x] == words[y]]

    if points:
        x,y=zip(*points)
    else:
        x=y=()

    img = BytesIO()
    plt.scatter(x,y)#,"r|",scalex=0.1)
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
    
def summary(n_sentences, text):
    words = nltk.word_tokenize(text)
    sentences = nltk.sent_tokenize(text)
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
                if len(sent.split(' ')) < 60 and len(sent.split(' ')) > 5:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(int(n_sentences), sentence_scores, key=sentence_scores.get)
    summary_text = ' '.join(summary_sentences)
    return summary_text


remove_punct_dict = dict((ord(punct),None) for punct in string.punctuation)
def LemNormalize(text1):
    return word_tokenize(text1.lower().translate(remove_punct_dict))

def ask_question():
    recognizer = sr.Recognizer() 
    mic = sr.Microphone(device_index=1) 
    with mic as source:
        print ('please say the phrases you like to know more about :')
        recognizer.adjust_for_ambient_noise(source) 
        captured_audio = recognizer.listen(source = mic)        
    
    text = recognizer.recognize_google(captured_audio)    
    return text

def response(typ, user_response, n_responses, text):
  n_responses = int(n_responses)
  sentences = nltk.sent_tokenize(text)
  if typ == 'voice':
      user_response = ask_question()
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
      robo_response = []
      c = 1
      i = 1
      while c <= n_responses :  
          idx = val.argsort()[0][-i]
          flat = val.flatten()
          flat.sort()
          score = flat[-i]
          m = nltk.word_tokenize(sentences[idx])
          if score > 0 and len(m) >= 6:
              robo_response.append(sentences[idx])
              c += 1
          i += 1 
          if score == 0:
              break
    
      if len(robo_response) == 0:
        robo_response = "sorry,I dont understand"
       
      sentences.remove(user_response)
  return robo_response


def pos_tag_plot (text):
     img = BytesIO()
     postag=[[pos_tag(word_tokenize(i))] for i in text.split('.')] 
     plt.figure(figsize = (8, 5), facecolor = None) 
     viz = PosTagVisualizer()
     viz.fit(postag)
     plt.ylabel('count')
     viz.show()
     plt.savefig(img,format='png')
     plt.close()
     img.seek(0)
     plot_url = base64.b64encode(img.getvalue()).decode('utf8')
     
     return plot_url
 
    
def frequency_chart(min_f, text):
    words = nltk.word_tokenize(text)
    min_f = int(min_f)
    img = BytesIO()
    sort_count = [[k, v] for k, v in sorted(dict(Counter(words)).items(),
                                                key=lambda item: item[1])]
    z=[j for j in sort_count if j[1] >= min_f][::-1]
    x,y = zip(*z)
    plt.figure(figsize = (25-min_f, 8))
    plt.bar(x,y)
    plt.xticks(rotation=45)
    plt.xlabel('most frequent words')
    plt.ylabel('count')
    plt.savefig(img,format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return plot_url    


