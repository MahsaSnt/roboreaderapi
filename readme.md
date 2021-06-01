### This api represents some features to analyze an article
#### routes and key values are mentioned below:

- https://roboreader.herokuapp.com/api, key vales : "type", "path"
   - Enter the type of your desired article ("url" or "text_file" or "audio_file") and the path of it, if it's type is "url" in path option insert the url of the article and if it's type is "file" in path option insert the directory of the file in the server, also the format of text_files should be "txt" or "docx" or "pdf" and the format of audio_files should be "mp3" or "wav". For example {"type": "url", "path": "https://en.wikipedia.org/wiki/Data_science"} or {"type": "text_file", "path": "D:/portfolio/roboreader/static/test.txt"}. This endpoint return the text of the article.


- https://roboreader.herokuapp.com/api/cleantext, key value : "text"
    - Enter the text of the article from api endpoint to get a dictionary including "clean_text" and "lemmatized_text". For example {"text": "..."}


- https://roboreader.herokuapp.com/api/wordcloud, key values : "min_font" , "lemmatized_text"
   - Enter the minimum font in wordcloud (which is directly related to the frequency of words) and lemmatized_text from api/cleantext endpoint to get the wordcloud of your text. For example {"min_font": "15", "lemmatized_text": "..."}
   

- https://roboreader.herokuapp.com/api/ngram, key values : "n_words", "n_ngrams", "lemmatized_text"
   - Enter number of the words which are most repeated beside each other (n_words), the number of these kind of sets (n_ngrams) and lemmatized_text from api/cleantext endpoint. For example {"n_words": "3", "n_ngrams": "5", "lemmatized_text": "..."}


- https://roboreader.herokuapp.com/api/dispersionplot, key values : "words", "lemmatized_text"
    - Enter a list of words which you like to check their dispersion in the text and lemmatized_text from api/cleantext endpoint. For example {"words": ["data", "analysis", "algorithm"], "lemmatized_text": "..."}


- https://roboreader.herokuapp.com/api/summary, key values : "n_sentences", "clean_text"
    - Enter number of sentences of the summary and clean_text from api/cleantext endpoint. For example {"n_sentences": "10", "clean_text": "..."}
    

- https://roboreader.herokuapp.com/api/response, key values : "type", "question", "clean_text"
    - Enter the type ("voice" or "text"), your question and clean_text from api/cleantext endpoint. For example {"type": "text", "question": "what is data mining", "clean_text": "..."}


- https://roboreader.herokuapp.com/api/postagplot, key value : "clean_text"
	- Enter clean_text from api/cleantext endpointit to get postag_plot of the text. For example {"clean_text": "..."}
	
	
- https://roboreader.herokuapp.com/api/frequencychart, key values : "min_frequency", "lemmatized_text"
	- Enter minimum frequency of words and lemmatized_text from api/cleantext endpoint to get a chart with the words which the frequency of them is at least equal to your number. For example {"min_frequency": "10", "lemmatized_text": "..."}
	
	
			