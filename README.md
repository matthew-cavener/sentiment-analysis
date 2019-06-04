# Quick Start
Clone this repo:

`git clone https://github.com/matthew-cavener/sentiment-analysis.git`

Navigate to `sentiment-analysis/server`:

`cd sentiment-analysis/server`

Start the training and server(It will take awhile):

`docker-compose up`

Use the `/predict` endpoint at `0.0.0.0:8080` to make sentiment predictions about Apple:
```
curl -X POST \
  http://0.0.0.0:8080/predict \
  -F 'text= My computer is so broken!'

returns:
{"not_relevant": 0.005387865395154152, "negative": 0.9436414900529212, "positive": 0.018740593702031454, "neutral": 0.03223005084989302}

curl -X POST \
  http://0.0.0.0:8080/predict \
  -F 'text=My computer is so awesome!'

returns:
{"not_relevant": 0.0020955397161594065, "negative": 0.02260511985303398, "positive": 0.9544383861491378, "neutral": 0.02086095428166866}

curl -X POST \
  http://0.0.0.0:8080/predict \
  -F 'text=My macbook is not a terribly bad machine.'

returns:
{"not_relevant": 0.0031329688599764367, "negative": 0.18629947141244943, "positive": 0.7409472185026398, "neutral": 0.06962034122493423}

curl -X POST \
  http://0.0.0.0:8080/predict \
  -F 'text=I wish apple could get their act together and stop blocking the whole screen with a volume indicator.'

returns:
{"not_relevant": 0.0005410340520799686, "negative": 0.9408746689195631, "positive": 0.0050036156983900865, "neutral": 0.05358068132996676}
```

Alternatively to `cURL`, use something like Postman.

# About
First things first: if you are wondering why it takes so long to startup the first time, it uses the large Transformer version of the [Universal Sentence Encoder.](https://tfhub.dev/google/universal-sentence-encoder-large/3)
So it has to download that module, and it is about 800MB. Then it trains the SVC used for classification which takes about 30 seconds. Mostly because of the whole not loading random pickles from the internet thing.
After the first start the USE module should be cached locally, so future training should be much faster. And it should have the classifier trianed, so you won't need to train it again. Future `up`s won't take nearly as long. Several seconds at worst. Plus inference is super fast. Like, half a second fast.

There are other classifiers just waiting to be trained if you want to use them. You can find them all in `server/models/train`. You can train them by running:
`docker-compose run sentiment python3 ./models/train/$classifier.py`

Replacing `$classifier` with the classifier you want to run, such as `BernoulliNB` or `RandomForest`. The ones that can be used for probability prediction will spit out pickles to the `pickles` directory.
 `AutoOpt.py` and `SVCBayOpt.py` don't work yet, pending my exploration of the libraries they use.

