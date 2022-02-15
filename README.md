# Sentiment Analysis App

[![Test on Python3.9](https://github.com/Tim-Abwao/sentiment-analysis-app/actions/workflows/run-tests.yml/badge.svg)](https://github.com/Tim-Abwao/sentiment-analysis-app/actions/workflows/run-tests.yml)

Perform [sentiment analysis][sentiment-analysis] using the traditional [bag of words][b-o-w] technique with [TF-IDF weighting][tfidf].

Powered by [Scikit-learn][sklearn] and [Streamlit][streamlit].

![Screen cast](screencast.gif)

The datasets used to train the models are the manually annotated samples from the publication *["From Group to Individual Labels Using Deep Features"][paper]*.

## Running Locally

1. Download the code, and create a virtual environment:

        git clone https://github.com/Tim-Abwao/sentiment-analysis-app.git
        cd sentiment-analysis-app
        python3 -m venv venv
        source venv/bin/activate

2. Install the required dependencies:

        pip install -U pip
        pip install -r requirements.txt

3. Launch the streamlit development server:

        streamlit run streamlit_app.py

[b-o-w]: https://en.wikipedia.org/wiki/Bag-of-words_model
[paper]: https://dl.acm.org/doi/10.1145/2783258.2783380 "Dimitrios Kotzias, Misha Denil, Nando de Freitas, and Padhraic Smyth. 2015. From Group to Individual Labels Using Deep Features. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '15). Association for Computing Machinery, New York, NY, USA, 597–606."
[sentiment-analysis]: https://en.wikipedia.org/wiki/Sentiment_analysis
[sklearn]: https://scikit-learn.org/
[streamlit]: https://streamlit.io/
[tfidf]: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
