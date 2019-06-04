import utils

from sklearn.model_selection import train_test_split
from hpsklearn import HyperoptEstimator, tfidf, any_sparse_classifier
from hyperopt import tpe

train, test = train_test_split(utils.read_data(), test_size=0.2)

# train_embeddings = utils.combined_embeddings(train['text'].tolist())
# test_embeddings = utils.combined_embeddings(test['text'].tolist())


estim = HyperoptEstimator(
    algo=tpe.suggest,
    max_evals=50,
    trial_timeout=300,
    preprocessing=[tfidf('tfidf')],
    classifier=any_sparse_classifier('clf')
)
estim.fit(train, train['sentiment'])

prediction = estim.predict(test['sentiment'])

score = estim.score(test, test['sentiment'])

model = estim.best_model()

print(score)
print(model)
