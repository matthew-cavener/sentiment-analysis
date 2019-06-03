import utils

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

train, test = train_test_split(utils.read_data(), test_size=0.2)

train_embeddings = utils.combined_embeddings(train['text'].tolist())
test_embeddings = utils.combined_embeddings(test['text'].tolist())

clf = RFC()
clf.fit(train_embeddings, train['sentiment'])

prediction = clf.predict(test_embeddings)
report = classification_report(test['sentiment'], prediction)
print(report)