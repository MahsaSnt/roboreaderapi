from flask import Flask, request, jsonify 
from flask_cors import CORS
from functions import word_cloud, n_gram, dispersion_plot, summary, response, clean_text, pos_tag_plot
from functions import frequency_chart, give_text
import numpy as np

app=Flask(__name__)
CORS(app)

@app.route('/api', methods=['POST'])
def index():
    data = request.get_json()
    typ = data["type"]
    path = data['path']
    min_silence_len = data['min_silence_len']
    directory = data['directory']
    text = give_text(typ, path, directory, int(min_silence_len))
    return jsonify(text= f"{text}")


@app.route('/api/cleantext', methods=['POST'])
def cl():
    data = request.get_json()
    text = data['text']
    clean_data = clean_text(text)
    clean = clean_data[0]
    lem = clean_data[1]
    return jsonify(clean_text= f"{clean}", lemmatized_text=f'{lem}')


@app.route('/api/wordcloud',methods=['POST'])
def wordcloud():
    data = request.get_json()
    n = data["min_font"]
    lem_text = data["lemmatized_text"]
    plot_url = word_cloud(n,lem_text)
    return jsonify(plot_url= f"{plot_url}")
    
    
@app.route('/api/ngram',methods=['POST'])
def ngram():
    data = request.get_json()
    n_words = data["n_words"]
    n_ngrams = data["n_ngrams"]
    lem_text = data["lemmatized_text"]
    plot_url = n_gram(n_words, n_ngrams, lem_text)
    return jsonify(plot_url= f"{plot_url}")
    

@app.route('/api/dispersionplot',methods=['POST'])
def dispersion():
    data = request.get_json()
    words = data["words"]
    lem_text = data["lemmatized_text"]
    plot_url = dispersion_plot(lem_text, words)
    
    return jsonify(plot_url= f"{plot_url}")
    
    
@app.route('/api/summary',methods=['POST'])
def summari():
    data = request.get_json()
    n_sentences = data["n_sentences"]
    text = data["clean_text"]
    summ = summary(n_sentences, text)
    return jsonify(Summary = f"{summ}")    


@app.route('/api/response',methods=['POST'])
def roboresponse():
    data = request.get_json()
    typ = data["type"]
    user_question = data["question"]
    text = data["clean_text"]
    n = data["n_responses"]
    res = response(typ, user_question, n, text)
    return jsonify(RoboResponse = res)   


@app.route ('/api/postagplot', methods=['POST'])
def pos():
    data = request.get_json()
    text = data["clean_text"]
    plot_url = pos_tag_plot(text)
    return jsonify(plot_url= f"{plot_url}")

    
    
@app.route ('/api/frequencychart', methods = ['POST'])
def fchart():
    data = request.get_json()
    min_f = data["min_frequency"]
    lem_text = data["lemmatized_text"]
    plot_url = frequency_chart (min_f, lem_text)
    
    return jsonify(plot_url = f"{plot_url}")


if __name__ == '__main__':
    app.run(debug = True)