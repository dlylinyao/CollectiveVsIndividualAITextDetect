from pathlib import Path

from einops import rearrange, repeat
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import make_scorer, accuracy_score
import matplotlib.pyplot as plt
import numpy as np


def build_dataset(stat_dir="stat", stat_name="*.csv"):
    stat_files = list(Path(stat_dir).glob(stat_name))
    stat_files.sort()

    X = []
    Y = []
    for stat_file in stat_files:
        text = stat_file.read_text(encoding="utf-8").strip()
        lines = text.split("\n")[1:]
        values = [
            [0.0 if v == "NA" else float(v) for v in l.strip().split(",")[1:]]
            for l in lines
        ]
        values = np.array(values)  # (l=75,c=5)
        positives = values[:, :1]
        negatives = values[:, 2::2]  # XXX [:, 1:] 
        # 1::2 positive+negative: 0.9425+/-0.0240
        # 2::2 positive+imitate: 0.8024+/-0.0554
        positives = repeat(positives, "l 1 -> l c", c=negatives.shape[1])
        X += [positives, negatives]
        Y += [np.ones(positives.shape[1]), np.zeros(negatives.shape[1])]
    X = np.concatenate(X, axis=1).astype("float32").transpose()
    Y = np.concatenate(Y, axis=0).astype("int32")

    return X, Y


def main():
    X, y = build_dataset("stat", "*.csv")
    print(X.shape, y.shape, X.max(), X.min())

    symlog = lambda _: np.sign(_) * np.log(1 + np.abs(_))
    # x = np.arange(-100, 100)
    # y = symlog(x)
    # plt.plot(x, y)
    # plt.show()
    X = symlog(X)

    # https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html
    classifiers = [
        # 0.9327+/-0.0346 0.7500+/-0.0576
        ["lr", LogisticRegression(C=5, max_iter=1000)],
        # 0.9326+/-0.0292 0.8270+/-0.0382
        ["rc", RidgeClassifier(alpha=1.0, solver="sparse_cg")],
        # 0.7095+/-0.1227 0.6875+/-0.0787
        ["knc", KNeighborsClassifier(n_neighbors=100)],
        # 0.8729+/-0.0670 0.7888+/-0.0759
        ["rfc", RandomForestClassifier()],
        # 0.9088+/-0.0532 0.7742+/-0.0661
        ["lsvc", LinearSVC(C=0.1, dual=False, max_iter=1000)],
        # 0.8100+/-0.1117 0.6155+/-0.1181
        [
            "sgdc",
            SGDClassifier(
                loss="log_loss", alpha=1e-4, n_iter_no_change=3, early_stopping=True
            ),
        ],
        # 0.8199+/-0.0499 0.6948+/-0.0430
        ["nc", NearestCentroid()],
        # 0.7164+/-0.0888 0.6394+/-0.0474
        ["cnb", ComplementNB(alpha=0.1)],
    ]
    # clf = VotingClassifier(classifiers)  # 0.7574+/-0.0738
    clf = StackingClassifier(classifiers)  # 0.8389+/-0.0249
    # model ensamble

    scores = cross_validate(clf, X, y, scoring=make_scorer(accuracy_score), cv=5)
    # print(scores)
    acc_mean = np.mean(scores["test_score"]).item()
    acc_std = np.std(scores["test_score"]).item()
    print(f"{acc_mean:.4f}+/-{acc_std:.4f}")
    return

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(pred)
    print(y_test)
    acc = accuracy_score(y_test, pred)
    print(acc)


if __name__ == "__main__":
    main()
