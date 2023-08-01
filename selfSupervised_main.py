import sys
import re
import os
import tarfile
import json

import numpy as np
from numpy.linalg import norm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from sklearn import preprocessing
import classify
import self_supervised

import nltk
from nltk import FreqDist
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

'''
Findings -
1) The default count vectorizer does not perform lemmatization.
2) stop words are not removed as well.
3) We can compare the performance of none, only stemmer, only lemmatizer, both stemmer and lemmatizer
'''


class Data:
    pass


def __data_modifier__(speech):
    stop_words = set(stopwords.words("english"))
    # This function transforms the input text data for classification
    # In addition to these we should check the hashing and tfidf vectorizer for performance.

    print("-- transforming data and labels")
    # speech.count_vect = CountVectorizer(
    #    max_df=0.9, preprocessor=custom_preprocessor)

    speech.count_vect = CountVectorizer(
        stop_words=stop_words, ngram_range=(1, 2))

    # speech.hash_vect = HashingVectorizer(
    #    stop_words=stop_words, preprocessor=custom_preprocessor)

    # speech.tfidf_vect = TfidfVectorizer(
    #    stop_words=stop_words, ngram_range=(1, 2))

    speech.trainX = speech.count_vect.fit_transform(speech.train_data)
    speech.devX = speech.count_vect.transform(speech.dev_data)

    speech.le = preprocessing.LabelEncoder()
    speech.le.fit(speech.train_labels)
    speech.target_labels = speech.le.classes_
    speech.trainy = speech.le.transform(speech.train_labels)
    speech.devy = speech.le.transform(speech.dev_labels)

    return


def read_files(tarfname):
    """Read the training and development data from the speech tar file.
    The returned object contains various fields that store the data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """

    tar = tarfile.open(tarfname, "r:gz")

    speech = Data()
    print("-- train data")
    speech.train_data, speech.train_fnames, speech.train_labels = read_tsv(
        tar, "train.tsv")
    print(len(speech.train_data))
    print("-- dev data")
    speech.dev_data, speech.dev_fnames, speech.dev_labels = read_tsv(
        tar, "dev.tsv")
    print(len(speech.dev_data))

    # This below function is used for transforming the input training data.
    __data_modifier__(speech)

    tar.close()
    return speech


def read_unlabeled(tarfname, speech):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the speech.vectorizer
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")

    class Data:
        pass
    unlabeled = Data()
    unlabeled.data = []
    unlabeled.fnames = []
    for m in tar.getmembers():
        if "unlabeled" in m.name and ".txt" in m.name:
            unlabeled.fnames.append(m.name)
            unlabeled.data.append(read_instance(tar, m.name))
    unlabeled.X = speech.count_vect.transform(unlabeled.data)
    print(unlabeled.X.shape)
    tar.close()
    return unlabeled


def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    fnames = []
    for line in tf:
        line = line.decode("utf-8")
        (ifname, label) = line.strip().split("\t")
        # print ifname, ":", label
        content = read_instance(tar, ifname)
        labels.append(label)
        fnames.append(ifname)
        data.append(content)
    return data, fnames, labels


def write_pred_kaggle_file(unlabeled, cls, outfname, speech):
    """Writes the predictions in Kaggle format.

    Given the unlabeled object, classifier, outputfilename, and the speech object,
    this function write the predictions of the classifier on the unlabeled data and
    writes it to the outputfilename. The speech object is required to ensure
    consistent label names.
    """
    yp = cls.predict(unlabeled.X)
    labels = speech.le.inverse_transform(yp)
    f = open(outfname, 'w')
    f.write("FileIndex,Category\n")
    for i in range(len(unlabeled.fnames)):
        fname = unlabeled.fnames[i]
        # iid = file_to_id(fname)
        f.write(str(i+1))
        f.write(",")
        # f.write(fname)
        # f.write(",")
        f.write(labels[i])
        f.write("\n")
    f.close()


def file_to_id(fname):
    return str(int(fname.replace("unlabeled/", "").replace("labeled/", "").replace(".txt", "")))


def write_gold_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the truth.

    You will not be able to run this code, since the tsvfile is not
    accessible to you (it is the test labels).
    """
    f = open(outfname, 'w')
    f.write("FileIndex,Category\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (ifname, label) = line.strip().split("\t")
            # iid = file_to_id(ifname)
            i += 1
            f.write(str(i))
            f.write(",")
            # f.write(ifname)
            # f.write(",")
            f.write(label)
            f.write("\n")
    f.close()


def write_basic_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the naive baseline.

    This baseline predicts OBAMA_PRIMARY2008 for all the instances.
    You will not be able to run this code, since the tsvfile is not
    accessible to you (it is the test labels).
    """
    f = open(outfname, 'w')
    f.write("FileIndex,Category\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (ifname, label) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write("OBAMA_PRIMARY2008")
            f.write("\n")
    f.close()


def read_instance(tar, ifname):
    inst = tar.getmember(ifname)
    ifile = tar.extractfile(inst)
    content = ifile.read().strip()
    return content


if __name__ == "__main__":
    # generateFolder()

    print("Reading data")
    tarfname = "./speech.tar.gz"
    speech = read_files(tarfname)

    print("Evaluating")

    print("Reading unlabeled data")
    unlabeled = read_unlabeled(tarfname, speech)

    speech.unlabelled_vectors = unlabeled.X.toarray()
    speech.unlabelled_text = unlabeled.data
    speech.unlabeled = unlabeled

    x = self_supervised.self_supervised_learner(speech)

    x.train()

    #
    # print("Writing pred file")
    # write_pred_kaggle_file(unlabeled, cls, "./speech-pred.csv", speech)

    # You can't run this since you do not have the true labels
    # print "Writing gold file"
    # write_gold_kaggle_file("data/speech-unlabeled.tsv", "data/speech-gold.csv")
    # write_basic_kaggle_file("data/speech-unlabeled.tsv", "data/speech-basic.csv")

    '''
# Idea

1) Do an analysis of each of the generated 19 documents. Find most common word. eg - P(obama | healthcare) should be high.
2) Certain speakers have a tendency to focus more on certain topics. Exploit that and add that information to the model.

'''
