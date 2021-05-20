from flask import Flask, request, jsonify
from flask_cors import CORS

from functions import word_cloud, n_gram, dispersion_plot, summary, response, clean_text, pos_tag_plot, mendenhall_curve
from functions import frequency_chart

app=Flask(__name__)
CORS(app)

 
@app.route('/api', methods=['POST'])
def index():
    data = request.get_json()
    url = data["url"]
    clean = clean_text(url)
    app.config['text'] = clean[0]
    app.config['sentences'] = clean[1]
    app.config['words'] = clean[2]
    app.config['lem_text'] = clean[3]
    app.config['lem_words'] = clean[4]
    app.config['postag'] = clean[5]
    status = 'success'
    return jsonify(status= f"{status}")

@app.route('/api/wordcloud',methods=['POST'])
def wordcloud():
    data = request.get_json()
    n = data["min_font"]
    text = app.config['lem_text']
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
    text = app.config['lem_text']
    plot_url = n_gram(n_words, n_ngrams, text)
    
    return jsonify(plot_url= f"{plot_url}")
    

@app.route('/api/dispersionplot',methods=['POST'])
def dispersion():
    data = request.get_json()
    words = data["words"]
    text = app.config['lem_text']
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


@app.route ('/api/postagplot', methods=['POST'])
def pos():
    text = app.config["text"]
    plot_url = pos_tag_plot(text)
    return jsonify(plot_url= f"{plot_url}")


@app.route ('/api/mendenhallcurve', methods=['POST'])
def mendenhall():
    words = app.config['words']
    plot_url = mendenhall_curve(words)
    return jsonify(plot_url= f"{plot_url}")
    
# @app.route ('/api/keywords', methods = ['POST'])
# def keyW():
#     data = request.get_json()
#     n = data["n_words"]
#     text = app.config['lem_text']
#     kw = key_words (text, n)
    
#     return jsonify(keywords= f"{kw}")

    
@app.route ('/api/frequencychart', methods = ['POST'])
def fchart():
    data = request.get_json()
    min_f = data["min_frequency"]
    words = app.config['lem_words']
    plot_url = frequency_chart (min_f, words)
    
    return jsonify(plot_url = f"{plot_url}")

    # url = 'https://en.wikipedia.org/wiki/Data_science'
    # min_f = 8
    # words = clean_text(url)[2]
    # plot_url = frequency_chart (min_f, words)
    # return render_template('plot.html', plot_url=plot_url, t='WordCloud') 


if __name__ == '__main__':
    app.run()