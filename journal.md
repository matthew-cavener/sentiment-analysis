# A place to record my thought process.

This isn't going to be formatted well. You have been warned.

So the problem is to create a server to do sentiment analysis, specifically using the apple twitter sentiment dataset here:

`https://www.figure-eight.com/wp-content/uploads/2016/03/Apple-Twitter-Sentiment-DFE.csv`

So in classic presentation form we'll begin by asking: What is sentiment analysis?
Well its analyzing sentiment, exactly what is says on the tin. But short of crowdsourcing, how do you technically implement it?
Well it seems like a multiclass classification problem at first glance, good, neutral, bad. Or more really, Happy, sad, frustrated, etc. like those emotion identification charts. Or it could be a numeric scale, [-1,1].

Looks like that is called categorical vs scalar. Neat.

So looking at the dataset, it is the multiclass, 1,3,5  categorical option. What do 1, 3, and 5 mean? The description of the dataset is:
```
Apple Computers Twitter sentiment

A look into the sentiment around Apple, based on tweets containing #AAPL, @apple, etc.

Contributors were given a tweet and asked whether the user was positive, negative, or neutral about Apple. (They were also allowed to mark “the tweet is not about the company Apple, Inc.)

Tweets cover a wide array of topics including stock performance, new products, IP lawsuits, customer service at Apple stores, etc.
```
That tells me what the numeric classes translate to but not their actual translations.
Well, judging by the text it seems 1 is bad, e.g. this tweet with a sentiment of 1 and confidence of 1.:

`WTF MY BATTERY WAS 31% ONE SECOND AGO AND NOW IS 29% WTF IS THIS @apple`

I'm willing to bet that means 3 is neutral and 5 is positive. Lets go test that hypothesis.

Here is a tweet with label 5 and confidence 1, though it seems like it only had 3 labellers.:

`RT @peterpham: Bought my @AugustSmartLock at the @apple store..pretty good logo match . can't wait to install it! http://t.co/z8VKMhbnR3`

Seems pretty positive.

Lets go looking to see if it is talked about elsewhere.

Here's a kaggle competition that outlines the labels:

`https://www.kaggle.com/c/apple-computers-twitter-sentiment2/data`

Cool, so it looks like 1 is negative, 3 is neutral, and 5 is positive.
So what are the other columns? Most seem pretty self-explainatory. It seems for some of the tweets they didn't get enough labellers (`_trusted_judgements` I assume?), hence the `_golden` boolean.
But they had to stop eventually, so hence the `finalized` state, I guess. Presumably there were lots more tweets that didn't even get that far.
Looks like other folks online are looking for an explaination and not finding one. I may look harder for an actual one later if it seems necessary, but this should be more than enough to get started.

Cool, so now lets just throw something together to see what a naive, simple thing can do.

So what do we need to do?

1.) Clean up the dataset. In the tweet text alone there's links and @mentions and RT (retweet?) labels. Those probably won't help our classifier and will at the very least make it more complicated.
I think I only need the sentiment label and cleaned text. I don't think I want to try and do things like spell check or expand acronyms, that sounds like a seperate probably more difficult problem. Just need a couple regexes to peel off that stuff.

2.) Create a feature vector from that text. BoW? Word vectors w/ pooling? Sentence embedding? Something completely different? Bag of words tends to be surprisingly good, but still not great. If I used a pretrained language model I can probably get much better results.
On that note, I kinda want want to play around with fine tuning BERT, but I also want to try on device inference. Lets start with something simple, go from there.

3.) Train classifier. Lots of options here. Naive Bayes, SVM, NN? Maybe try 'em all?

So I think for my first go I want to use my ol' reliable featurizer, the Universal Sentence Encoder from TF-Hub. Though `ntlk`'s `VADER` module seems task built for this, so maybe I'll tack its results onto the feature. And since the embeddings are contextual, I don't want to use naive Bayes, since the features are very explicitly NOT independent. SVM w/ non linear kernel it is then.

I'll want to do a fair bit of experimentation with different featurizers and classifiers and don't want to be repeating a bunch of code, so I'll need a way to pass which of those I want to use in.

The problem is to create a sentiment analyzer using this dataset, so it feels like cheating if I went and grabbed a couple of other datasets.
I could probably use a few other twitter datasets to finetune my featurizer, and just use this dataset for the classifier. That seems well within the spirit of the problem. I'll probably do that later, but for now, I know what I need to do. Let's go do it!


Okay, so that's done, results of the 3 runs:
```
              precision    recall  f1-score   support

          -1       0.00      0.00      0.00        11
           1       0.87      0.76      0.81       259
           3       0.78      0.95      0.86       426
           5       0.84      0.38      0.52        82

    accuracy                           0.81       778
   macro avg       0.62      0.52      0.55       778
weighted avg       0.81      0.81      0.79       778

              precision    recall  f1-score   support

          -1       0.00      0.00      0.00        12
           1       0.79      0.79      0.79       229
           3       0.80      0.91      0.85       456
           5       0.81      0.31      0.45        81

    accuracy                           0.80       778
   macro avg       0.60      0.50      0.52       778
weighted avg       0.79      0.80      0.78       778

              precision    recall  f1-score   support

          -1       0.00      0.00      0.00        12
           1       0.80      0.74      0.77       168
           3       0.77      0.92      0.84       333
           5       0.85      0.31      0.46        70

    accuracy                           0.78       583
   macro avg       0.60      0.50      0.52       583
weighted avg       0.77      0.78      0.76       583
```

Using just the `VADER` "embeddings":
```
  'precision', 'predicted', average, warn_for)
              precision    recall  f1-score   support

           1       0.60      0.57      0.59       254
           3       0.65      0.79      0.71       423
           5       0.83      0.17      0.29        87
not_relevant       0.00      0.00      0.00        14

    accuracy                           0.64       778
   macro avg       0.52      0.38      0.40       778
weighted avg       0.64      0.64      0.61       778
```
Surprisingly good.

So the not_relevent tag is having a struggle. Let's try a NB.
First run doesn't look too promising, though it was ACTUALLY able to get a tiny bit of the `not_relevant` class.:
```
              precision    recall  f1-score   support

           1       0.72      0.73      0.73       261
           3       0.77      0.51      0.61       419
           5       0.31      0.47      0.37        75
not_relevant       0.10      0.52      0.17        23

    accuracy                           0.58       778
   macro avg       0.47      0.56      0.47       778
weighted avg       0.69      0.58      0.61       778
```

Bumping the `alpha` parameter up helps! `alpha=1000`:
```
              precision    recall  f1-score   support

           1       0.72      0.76      0.74       243
           3       0.79      0.75      0.77       443
           5       0.38      0.35      0.36        77
not_relevant       0.00      0.00      0.00        15

    accuracy                           0.70       778
   macro avg       0.47      0.47      0.47       778
weighted avg       0.71      0.70      0.71       778
```

Let's keep going, what's 1000 do?
```
              precision    recall  f1-score   support

           1       0.71      0.52      0.60       270
           3       0.64      0.90      0.75       408
           5       0.43      0.04      0.07        74
not_relevant       0.00      0.00      0.00        26

    accuracy                           0.66       778
   macro avg       0.45      0.37      0.36       778
weighted avg       0.62      0.66      0.61       778
```

Too much. Playing around some more, 50 seems like a good alpha.
Next a Random Forest.
Looks, well, okay I guess.
```
              precision    recall  f1-score   support

           1       0.67      0.71      0.69       228
           3       0.75      0.83      0.79       447
           5       0.59      0.24      0.34        83
not_relevant       0.20      0.05      0.08        20

    accuracy                           0.71       778
   macro avg       0.55      0.46      0.47       778
weighted avg       0.69      0.71      0.69       778
```
So far it seems like my intuition about the SVM is true, so I guess that's a good sign.
Let's try an ExtraTreesClassifier with 10 estimators:
```
              precision    recall  f1-score   support

           1       0.65      0.73      0.69       233
           3       0.73      0.84      0.78       425
           5       0.65      0.17      0.27        98
not_relevant       0.00      0.00      0.00        22

    accuracy                           0.70       778
   macro avg       0.51      0.44      0.44       778
weighted avg       0.67      0.70      0.67       778
```
More estimators is going to be wayyyy better, right? Well, turns out 1000 is about as good as 100, so...
What's 100000 do? Apparently take way longer and use a tone more memory. Okay, so it breaks.

KNN is surprisingly good, playing around with `n_neighbors` and the distance metric don't seem to do much:
```
              precision    recall  f1-score   support

           1       0.71      0.65      0.68       243
           3       0.72      0.82      0.77       425
           5       0.59      0.45      0.51        95
not_relevant       0.20      0.07      0.10        15

    accuracy                           0.71       778
   macro avg       0.56      0.50      0.51       778
weighted avg       0.69      0.71      0.70       778
```

So even a Linear SVC works pretty well:
```
              precision    recall  f1-score   support

           1       0.85      0.77      0.81       237
           3       0.77      0.93      0.84       431
           5       0.77      0.37      0.50        92
not_relevant       0.00      0.00      0.00        18

    accuracy                           0.79       778
   macro avg       0.60      0.52      0.54       778
weighted avg       0.78      0.79      0.77       778
```

The multilayer perceptron works pretty well out of the box, let's play with that.
So with 32 layers of 512 neurons, it... stayed about the same.
```
              precision    recall  f1-score   support

           1       0.82      0.73      0.77       246
           3       0.74      0.86      0.80       417
           5       0.65      0.48      0.55        98
not_relevant       0.29      0.12      0.17        17

    accuracy                           0.75       778
   macro avg       0.62      0.55      0.57       778
weighted avg       0.75      0.75      0.74       778
```
If I take the conventional wisdom, off 1 layer with a number of neurons between the input size and output size ~256:
```
              precision    recall  f1-score   support

           1       0.77      0.72      0.75       247
           3       0.77      0.84      0.80       426
           5       0.66      0.53      0.59        91
not_relevant       0.00      0.00      0.00        14

    accuracy                           0.75       778
   macro avg       0.55      0.52      0.53       778
weighted avg       0.74      0.75      0.75       778
```

So far the SVC performs best, which is kinda what I expected, cool.
![We can pickle that!](http://gph.is/2ht7ISo)

So I looked into some hyperparameter optimization methods, looks like there is a neat python library for it, `hyperopt-sklearn`, or more generall `hyperopt`.
The promise of hpsklearn was optimize all your hyperparameters, including your classifier choice. I could get it working after spending a bit with it, but I'll check it out later.
Then I tried just plain `hyperopt` on the SVC, it takes about 30 seconds per trial, and doesn't seem to be doing any better so far than the slightly tuned one I used before.
I'll procedd by just pickling up that SVC model, and creating an endpoint around `predict_proba()` with it. And probably a validation endpoint to make that step easy.
I'd LIKE to try out some other embeddings/fine tune another model, and maybe set up an ensemble to better predict that `not_relevant` class. I'll see how that goes.