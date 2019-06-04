import utils
import pickle

from os.path import isfile

from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

filename = '/usr/src/app/sentiment/models/pickles/MLPC.pickle'

if isfile(filename) == False:

    train, test = train_test_split(utils.read_data(), test_size=0.2)

    train_embeddings = utils.combined_embeddings(train['text'].tolist())
    test_embeddings = utils.combined_embeddings(test['text'].tolist())

    clf = MLPC(
        hidden_layer_sizes=(256),
        learning_rate='adaptive',
        max_iter=1000
    )
    clf.fit(train_embeddings, train['sentiment'])

    prediction = clf.predict(test_embeddings)
    report = classification_report(test['sentiment'], prediction)
    print(report)

    with open(filename, 'wb') as f:
        pickle.dump(clf, f)

else:
    print('Already Trained!')
