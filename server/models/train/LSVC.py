import utils

from sklearn.svm import LinearSVC as LSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

train, test = train_test_split(utils.read_data(), test_size=0.2)

train_embeddings = utils.combined_embeddings(train['text'].tolist())
test_embeddings = utils.combined_embeddings(test['text'].tolist())

clf = LSVC(
    multi_class='crammer_singer',
    C=1.0
)
clf.fit(train_embeddings, train['sentiment'])

prediction = clf.predict_proba(test_embeddings)[:, 1]
report = classification_report(test['sentiment'], prediction)
print(report)