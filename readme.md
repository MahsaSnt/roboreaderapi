### This api represents some features to analyze an article
#### routes and key values are mentioned below:

- https://roboreader.herokuapp.com/api, key vale : "url" 
   - enter the url of your desired article, for example {"url": "https://en.wikipedia.org/wiki/Data_science"}


- https://roboreader.herokuapp.com/api/wordcloud, key value : "min_font" 
   - enter the minimum font in wordcloud (which is directly related to the frequency of words) to get the wordcloud of your text, for example {"min_font": "15"}
   

- https://roboreader.herokuapp.com/api/ngram, key values : "n_words", "n_ngrams" 
   - enter number of the words which are most repeated beside each other (n_words) and the number of these kind of sets (n_ngrams),  for example {"n_words": "3", "n_ngrams": "5"}


- https://roboreader.herokuapp.com/api/dispersionplot, key value : "words"
    - enter a list of words which you like to check their dispersion in the text, for example {"words": ["data", "analysis", "algorithm"]}


- https://roboreader.herokuapp.com/api/summary, key value : "n_sentences"
    - enter number of sentences of the summary, for example {"n_sentences": "10"}
    

- https://roboreader.herokuapp.com/api/response, key value : "question"
    - enter your question from the text, for example {"question": "what is data mining"}

