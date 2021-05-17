from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from newspaper import Article
import string
import warnings
import nltk
import re
from nltk.corpus import stopwords


from functions import word_cloud, n_gram, dispersion_plot, summary, response


#ignore the warnings
warnings.filterwarnings('ignore')

#download package from nltk
nltk.download('punkt',quiet=True)
nltk.download('wordnet',quiet=True)
nltk.download('stopwords')


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


app=Flask(__name__)
CORS(app)

 
@app.route('/api', methods=['POST'])
def index():
    data = request.get_json()
    url = data["url"]
    app.config['text'] = clean_text(url)[0]
    app.config['sentences'] = clean_text(url)[1]
    app.config['words'] = clean_text(url)[2]
    status = 'success'
    return jsonify(status= f"{status}")

@app.route('/api/wordcloud',methods=['POST', 'GET'])
def wordcloud():
    data = request.get_json()
    n = data["min_font"]
    text = app.config['text']
    # url = 'https://en.wikipedia.org/wiki/Data_science'
    # n = '10'
    plot_url = word_cloud(n,text)
    return jsonify(plot_url= f"{plot_url}")

    # return render_template('plot.html', plot_url=plot_url, t='WordCloud') 
    
@app.route('/api/ngram',methods=['POST', 'GET'])
def ngram():
    data = request.get_json()
    n_words = data["n_words"]
    n_ngrams = data["n_ngrams"]
    text = app.config['text']
    plot_url = n_gram(n_words, n_ngrams, text)
    
    return jsonify(plot_url= f"{plot_url}")
    

@app.route('/api/dispersionplot',methods=['POST', 'GET'])
def dispersion():
    data = request.get_json()
    words = data["words"]
    text = app.config['text']
    plot_url = dispersion_plot(text, words)
    
    return jsonify(plot_url= f"{plot_url}")

@app.route('/api/summary',methods=['POST', 'GET'])
def summari():
    data = request.get_json()
    n_sentences = data["n_sentences"]
    sentences = app.config['sentences']
    words = app.config['words']
    summ = summary(n_sentences, sentences, words)
    return jsonify(Summary = f"{summ}")    


@app.route('/api/response',methods=['POST', 'GET'])
def roboresponse():
    data = request.get_json()
    user_question = data["question"]
    sentences = app.config['sentences']
    
    res = response(user_question, sentences)
    return jsonify(RoboResponse = f"{res}")   


if __name__ == '__main__':
    app.run(debug=True)