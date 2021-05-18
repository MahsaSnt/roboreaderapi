from flask import Flask, request, jsonify
from flask_cors import CORS

from functions import word_cloud, n_gram, dispersion_plot, summary, response, clean_text


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

@app.route('/api/wordcloud',methods=['POST'])
def wordcloud():
    data = request.get_json()
    n = data["min_font"]
    text = app.config['text']
    # url = 'https://en.wikipedia.org/wiki/Data_science'
    # n = '10'
    plot_url = word_cloud(n,text)
    return jsonify(plot_url= f"{plot_url}")

    # return render_template('plot.html', plot_url=plot_url, t='WordCloud') 
    
@app.route('/api/ngram',methods=['POST'])
def ngram():
    data = request.get_json()
    n_words = data["n_words"]
    n_ngrams = data["n_ngrams"]
    text = app.config['text']
    plot_url = n_gram(n_words, n_ngrams, text)
    
    return jsonify(plot_url= f"{plot_url}")
    

@app.route('/api/dispersionplot',methods=['POST'])
def dispersion():
    data = request.get_json()
    words = data["words"]
    text = app.config['text']
    plot_url = dispersion_plot(text, words)
    
    return jsonify(plot_url= f"{plot_url}")

@app.route('/api/summary',methods=['POST'])
def summari():
    data = request.get_json()
    n_sentences = data["n_sentences"]
    sentences = app.config['sentences']
    words = app.config['words']
    summ = summary(n_sentences, sentences, words)
    return jsonify(Summary = f"{summ}")    


@app.route('/api/response',methods=['POST'])
def roboresponse():
    data = request.get_json()
    user_question = data["question"]
    sentences = app.config['sentences']
    
    res = response(user_question, sentences)
    return jsonify(RoboResponse = f"{res}")   


if __name__ == '__main__':
    app.run()