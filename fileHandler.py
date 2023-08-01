import sys
import re
import os
import tarfile
import json
import time
import seaborn as sns

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE

from sklearn import preprocessing
import classify
import self_supervised

import nltk
from nltk import FreqDist
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# What happens if the test dataset has words that are not present in the training dataset? how does the count vectorizer work? wont there be a dimension mismatch?

# nltk.download('punkt')
# nltk.download('wordnet')

'''Documenting the current results'''
# The initial default training accuracy is - 99.26 (without any changes)
# The initial default test accuracy is - 41.30 (without any changes)

# The training accuracy after removing stop words alone is - 98.74
# The test accuracy after removing stop words alone is - 40.33

# The training accuracy after removing stop words and using Porter Stemmer is - 97.16
# The test accuracy after removing stop words and using Porter Stemmer is - 37.92

# The training accuracy after removing stop words and using lemmatizer is - 98.30
# The test accuracy after removing stop words and using lemmatizer is - 40.33

# The training accuracy after removing stop words and using both Porter Stemmer and lemmatizer is - 97.16
# The test accuracy after removing stop words and using both Porter Stemmer and lemmatizer is - 38.16

'''
Findings -
1) The default count vectorizer does not perform lemmatization.
2) stop words are not removed as well.
3) We can compare the performance of none, only stemmer, only lemmatizer, both stemmer and lemmatizer


'''


class Data:
    pass


def hidden_state_visualizer(X, y, num_of_clusters, N=10000):
    feat_cols = ['pixel'+str(i) for i in range(X.shape[1])]
    print(X.shape)
    print(len(feat_cols))
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    X, y = None, None
    # print('Size of the dataframe: {}'.format(df.shape))

    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])

    df_subset = df.loc[rndperm[:N], :].copy()
    data_subset = df_subset[feat_cols].values

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", num_of_clusters),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    plt.show()
    return


def custom_preprocessor(text):
    # init stemmer
    porter_stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    text = re.sub("\\W", " ", text)  # remove special chars

    # stem words
    words = re.split("\\s+", text)
    stemmed_words = [porter_stemmer.stem(word=word) for word in words]

    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)  # .join(stemmed_words)  #


def test_vectorizer():
    stop_words = set(stopwords.words("english"))
    text = [
        "John is a good boy. John watches basketball and is also watching cricket.the cow"]

    vectorizer = CountVectorizer(
        stop_words=stop_words, preprocessor=custom_preprocessor)

    vectorizer = HashingVectorizer(
        stop_words=stop_words, preprocessor=custom_preprocessor)

    # vectorizer = TfidfVectorizer(stop_words=stop_words, preprocessor=custom_preprocessor)
    # tokenize and build vocab
    vectorizer.fit(text)
    # print(vectorizer.vocabulary_)
    vector = vectorizer.transform(text)
    # print(vector.toarray())
    return


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


def write_pred_kaggle_file(yp, unlabeled, outfname, speech):
    """Writes the predictions in Kaggle format.

    Given the unlabeled object, classifier, outputfilename, and the speech object,
    this function write the predictions of the classifier on the unlabeled data and
    writes it to the outputfilename. The speech object is required to ensure
    consistent label names.
    """
    # yp = cls.predict(unlabeled.X)
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


def custom_data_handler(speech, ncomp):
    # print(speech)
    # speech.train_fnames, speech.train_labels
    print()
    print("Entered the custom Data Handler")
    print()
    print(speech.trainX.toarray().shape)

    # print(speech.count_vect.get_feature_names_out())
    # print(len(speech.count_vect.vocabulary_))

    # checking the cosine similarity between the vectors of two speeches by the same president
    # tfidf_transformer = TfidfTransformer()
    # speech.trainX = tfidf_transformer.fit_transform(speech.trainX)

    pca = PCA(n_components=ncomp)

    trainX = pca.fit_transform(speech.trainX.toarray())
    devX = pca.transform(speech.devX.toarray())

    # speech.trainX = speech.trainX.toarray()
    # speech.devX = speech.devX.toarray()

    # y = np.array(speech.trainy)
    # ref = y[12]
    # A = speech.trainX[ref]
    # norm_A = norm(A)
    # print(f"The reference label is {ref}")
    '''
    for i in range(y.shape[0]):
        if (i >= 100):
            sys.exit()

        if (y[i] == ref):
            print(f"{i},{y[i]}")
            B = speech.trainX[i, :]
            cosine = np.dot(A, B)/(norm_A*norm(B))
            print("Cosine Similarity:", cosine)
            print()
    '''
    return trainX, devX, pca


def generateFolder():
    problems_path = "./allSpeeches"
    isExist = os.path.exists(problems_path)
    if (not isExist):
        os.mkdir(problems_path)

    return


def create_all_speeches(speech):
    all_speeches = {}
    for i in range(len(speech.train_labels)):
        president = speech.train_labels[i]
        # print(speech.train_data[i])
        # sys.exit()
        if (president not in all_speeches):
            all_speeches[president] = speech.train_data[i] + b" "
        else:
            all_speeches[president] += speech.train_data[i] + b" "

    for key in all_speeches:
        # fdist1 = FreqDist(all_speeches[key].split())
        # print(f"For the label {key} the most common words are")
        # print(fdist1.most_common())
        # print()
        with open(f"./allSpeeches/{key}.txt", "wb") as f:
            f.write(all_speeches[key])

    return
# what instances it is working on and what instances it is failing on. Is the label ditribution very different. What is going wrong.
# what samples are added to the training dataset each epoch? Are they from the same label/class


def display_all_documents_similarity():
    stop_words = set(stopwords.words("english"))
    text_files = os.listdir("./allSpeeches")
    documents = [open(f"./allSpeeches/{speech}").read()
                 for speech in text_files]
    tf_idf_vectorizer = TfidfVectorizer(
        stop_words=stop_words, preprocessor=custom_preprocessor, ngram_range=(1, 2))
    tfidf_documents = tf_idf_vectorizer.fit_transform(documents)

    # print(tfidf_documents.shape)
    # pairwise_similarity = tfidf_documents * tfidf_documents.T
    # print(pairwise_similarity.toarray().shape)
    return tf_idf_vectorizer, tfidf_documents, text_files


def compute_cosine_similarity_based_accuracy(y, y_pred):
    acc = metrics.accuracy_score(y, y_pred)
    print("Accuracy :", acc)
    return


def compute_cosine_similarity_matrix(tf_idf_vectorizer, tfidf_documents, X, txt_files):
    # print(txt_files[0])
    y_pred = []
    # print(speech.dev_labels[0])
    cosine_sim_matrix = np.zeros((len(X), 19))
    for j in range(len(X)):
        # print(doc)
        doc_vec = tf_idf_vectorizer.transform([X[j]]).toarray()
        # print(doc_vec.shape)
        # print(tfidf_documents.shape)
        max_cosine_sim = 0
        pred_label = ''
        for i in range(tfidf_documents.shape[0]):
            # print(txt_files[i])
            # print(doc_vec.shape, tfidf_documents[i].shape)
            cosine_sim = np.dot(
                doc_vec, tfidf_documents[i].toarray().reshape((-1, 1)))[0][0]

            cosine_sim_matrix[j][i] = cosine_sim
            if (cosine_sim > max_cosine_sim):
                max_cosine_sim = cosine_sim
                pred_label = txt_files[i][:-4]

            # print(f"The cosine similarity is {cosine_sim}")
            # print()
        y_pred.append(pred_label)

    return cosine_sim_matrix, y_pred


def modify_data(train_data, cosine_sim_matrix):
    # In this function the training data is concatenated with the cosine similarity matrix
    try:
        train_data = np.array(train_data.toarray())  # .toarray()
    except:
        train_data = np.array(train_data)

    new_train_data = np.concatenate(
        (train_data, cosine_sim_matrix), axis=1)

    return new_train_data


def plot_acc_trends(train_acc_lst, val_acc_lst):
    y1 = train_acc_lst
    x1 = [item for item in range(0, len(y1))]

    # plotting the line 1 points
    plt.plot(x1, y1, label="Train Accuracy")

    # line 2 points
    y2 = val_acc_lst
    x2 = [item for item in range(0, len(y2))]

    # plotting the line 2 points
    plt.plot(x2, y2, label="Dev Accuracy")

    # naming the x axis
    plt.xlabel('Epochs')
    # naming the y axis
    plt.ylabel('Accuracy')
    # giving a title to my graph
    plt.title('Accuracy Vs Epochs')

    # show a legend on the plot
    plt.legend()
    plt.grid()
    # function to show the plot
    plt.show()
    return


if __name__ == "__main__":
    # generateFolder()
    pca_vals = [5, 10, 20, 50, 100, 200, 400, 800, 1600, 3200, 4000]
    train_acc_vals = []
    val_acc_vals = []

    tf_idf_vectorizer, tfidf_documents, text_files = display_all_documents_similarity()
    # test_vectorizer()
    # sys.exit()

    print("Reading data")
    tarfname = "./speech.tar.gz"
    speech = read_files(tarfname)

    print(speech.trainX.shape)

    print(speech.devX.shape)
    total_num_sample = 0

    # create_all_speeches(speech)

    # Below lines are there to coompute the cosine similarity between the speech documents and the queries.
    cosine_sim_matrix_train, y_pred = compute_cosine_similarity_matrix(
        tf_idf_vectorizer, tfidf_documents, speech.train_data, text_files)

    compute_cosine_similarity_based_accuracy(speech.train_labels, y_pred)

    cosine_sim_matrix_test, y_pred = compute_cosine_similarity_matrix(
        tf_idf_vectorizer, tfidf_documents, speech.dev_data,  text_files)

    compute_cosine_similarity_based_accuracy(speech.dev_labels, y_pred)

    # for val in pca_vals:
    # print(f"The current value of PCA is {val}")
    # trainX, devX, pca = custom_data_handler(speech, 4370)
    # hidden_state_visualizer(trainX,speech.train_labels, 19, N=30000)
    # sys.exit()
    # after the data is read. Calling the data handler to understand what the data looks like.

    print("Training classifier")
    # standardizing the training data
    # speech.trainX = speech.trainX/speech.trainX.max()
    # speech.devX = speech.devX/speech.devX.max()

    new_train_data = modify_data(speech.trainX, cosine_sim_matrix_train)
    new_dev_data = modify_data(speech.devX, cosine_sim_matrix_test)

    '''
    neigh = KNeighborsClassifier(n_neighbors=7)
    kmeans = KMeans(n_clusters=19, random_state=0)
    kmeans.fit(cosine_sim_matrix_train)


    knn = KNeighborsClassifier()
    from sklearn.model_selection import GridSearchCV
    k_range = list(range(1, 100))
    param_grid = dict(n_neighbors=k_range)

    # defining parameter range
    grid = GridSearchCV(knn, param_grid, cv=10,
                        scoring='accuracy', return_train_score=False, verbose=1)

    # fitting the model for grid search
    grid_search = grid.fit(cosine_sim_matrix_train,  speech.trainy)

    print(grid_search.best_params_)

    accuracy = grid_search.best_score_ * 100
    print(
        "Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))


    neigh.fit(cosine_sim_matrix_train,  speech.trainy)
    y_pred_knn = neigh.predict(cosine_sim_matrix_train)
    y_pred_knn1 = neigh.predict(cosine_sim_matrix_test)

    # new_train_data = np.concatenate(
    #    (new_train_data, y_pred_knn.reshape((-1, 1))), axis=1)

    # new_dev_data = np.concatenate(
    #    (new_dev_data, y_pred_knn1.reshape((-1, 1))), axis=1)

    y_pred_km = kmeans.predict(cosine_sim_matrix_train)
    y_pred_km_dev = kmeans.predict(cosine_sim_matrix_test)

    temp = {}
    for i in range(len(speech.trainy)):
        if (speech.trainy[i] in temp):
            temp[speech.trainy[i]].append(int(y_pred_km[i]))
        else:
            temp[speech.trainy[i]] = [int(y_pred_km[i])]

    label_converter = {}
    completed_labels = set()
    for key in temp:
        freq = {}
        for num in temp[key]:
            if (num in freq):
                freq[num] += 1
            else:
                freq[num] = 1

        print()
        keys = list(freq.keys())
        values = list(freq.values())
        sorted_value_index = np.argsort(values)
        sorted_dict = {keys[i]: values[i] for i in sorted_value_index}
        print(sorted_dict)
        print(f"{key} -> {max(sorted_dict, key=sorted_dict.get)}")
        label_converter[max(sorted_dict, key=sorted_dict.get)] = key

        if (max(sorted_dict, key=sorted_dict.get) not in completed_labels):
            label_converter[max(sorted_dict, key=sorted_dict.get)] = key
            completed_labels.add(max(sorted_dict, key=sorted_dict.get))
        else:
            label_converter[1] = 1
            completed_labels.add(1)

    label_converter[0] = 10
    label_converter[1] = 4

    print(label_converter)

    updated_pred_kmeans_train = []
    updated_pred_kmeans_test = []
    for val in y_pred_km:
        updated_pred_kmeans_train.append(label_converter[val])

    for val in y_pred_km_dev:
        updated_pred_kmeans_test.append(label_converter[val])

    acc = metrics.accuracy_score(speech.trainy, updated_pred_kmeans_train)
    print("Accuracy :", acc)

    acc = metrics.accuracy_score(speech.devy, updated_pred_kmeans_test)
    print("Accuracy :", acc)

    '''
    cls = classify.train_classifier(new_train_data, speech.trainy)

    print("Evaluating")
    train_acc = classify.evaluate(new_train_data, speech.trainy, cls)
    val_acc = classify.evaluate(new_dev_data, speech.devy, cls)

    train_acc_vals.append(train_acc)
    val_acc_vals.append(val_acc)

    # print(metrics.accuracy_score(speech.trainy, y_pred_knn))
    # print(metrics.accuracy_score(speech.devy, y_pred_knn1))
    # The below portion is used for self_supervised learning

    # The below portion deals with predicting for the unlabelled dataset and also serves as a starting point for self-supervised

    print("Reading unlabeled data")
    unlabeled = read_unlabeled(tarfname, speech)

    speech.unlabelled_vectors = unlabeled.X.toarray()
    speech.unlabelled_text = unlabeled.data
    print(speech.unlabelled_vectors.shape)

    cosine_sim_matrix_unlabelled, y_cosine_pred = compute_cosine_similarity_matrix(
        tf_idf_vectorizer, tfidf_documents, speech.unlabelled_text, text_files)

    # pca = PCA(n_components=4370)

    print("Writing pred file")
    unlabelled_data = modify_data(
        speech.unlabelled_vectors, cosine_sim_matrix_unlabelled)

    y_pred = cls.predict(unlabelled_data)
    write_pred_kaggle_file(y_pred, unlabeled, "./speech-pred1.csv", speech)

    # You can't run this since you do not have the true labels
    # print "Writing gold file"
    # write_gold_kaggle_file("data/speech-unlabeled.tsv", "data/speech-gold.csv")
    # write_basic_kaggle_file("data/speech-unlabeled.tsv", "data/speech-basic.csv")

    '''
# Idea

1) Do an analysis of each of the generated 19 documents. Find most common word. eg - P(obama | healthcare) should be high.
2) Certain speakers have a tendency to focus more on certain topics. Exploit that and add that information to the model.

'''
