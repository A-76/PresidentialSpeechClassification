from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from sklearn import metrics


import re
import os

import sys
import re
import os
import tarfile

import numpy as np
from numpy.linalg import norm

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn import metrics

from sklearn import preprocessing


import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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


def modify_data(train_data, cosine_sim_matrix):
    # In this function the training data is concatenated with the cosine similarity matrix
    # train_data = np.array(train_data)  # .toarray()
    new_train_data = np.concatenate(
        (train_data, cosine_sim_matrix), axis=1)

    return new_train_data


class self_supervised_learner():
    epsilon = 0.001  # this is the stopping criterion
    max_epochs = 10
    threshold = 0.90

    val_acc_lst = []
    train_acc_lst = []

    def write_to_file(self):
        return

    def plot_acc_trends(self):
        y1 = self.train_acc_lst
        x1 = [item for item in range(0, len(y1))]

        # plotting the line 1 points
        plt.plot(x1, y1, label="Train Accuracy")

        # line 2 points
        y2 = self.val_acc_lst
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

    def __init__(self, speech):
        self.x_train = speech.trainX.toarray()
        self.x_val = speech.devX.toarray()
        self.x_unlabelled = speech.unlabelled_vectors
        self.unlabeled = speech.unlabeled

        self.y_train = speech.trainy
        self.y_val = speech.devy

        self.unlabelled_text = speech.unlabelled_text
        self.labelled_train = speech.train_data
        self.labelled_dev = speech.dev_data
        self.le = speech.le

        self.model = LogisticRegression(
            warm_start=True, solver="liblinear", penalty="l2", C=2.2212)
        self.rfc = RandomForestClassifier()
        self.svc = SVC()
        self.mnb = MultinomialNB()
        self.knn = KNeighborsClassifier(n_neighbors=7)

        self.shifted_samples = {}
        self.is_completed = False

        # keep track of the values
        self.prev_val_acc = 0
        self.number_times_acc_decrease = 0
        self.max_added_samples = 0.01
        return
    # use a set for the removing samples part to make it more efficient

    def __remove_entries_from_unlabelled__(self, class_confidence):
        # we need to clean up both x_unlabelled as well as its correponding cosine_matrix
        i = 0
        num_deleted_samples = 0
        while (i < len(class_confidence)):
            if (class_confidence[i] == 1):
                self.x_unlabelled = np.delete(self.x_unlabelled, i, 0)
                self.cosine_sim_matrix_unlabelled = np.delete(
                    self.cosine_sim_matrix_unlabelled, i, 0)
                class_confidence = np.delete(class_confidence, i, 0)
                num_deleted_samples += 1
            else:
                i += 1
        # print(f"{num_deleted_samples} have been moved from unlabeled to labeled.")
        return

    def __update_training_data__(self, class_confidence, corresponding_class):
        print("Trying to update the training data")
        if (not sum(class_confidence)):
            self.is_completed = True
            return
        additions_x = np.zeros(
            (int(sum(class_confidence)), self.x_unlabelled.shape[1]))

        additions_y = np.zeros((int(sum(class_confidence)), 1))

        additions_cosine = np.zeros((int(sum(class_confidence)), 19))
        j = 0
        for i in range(len(class_confidence)):
            if (class_confidence[i] == 1):
                # Since we initialized with 0, adding and assigning are equivalent
                # self.shifted_samples.add(i)
                additions_x[j] += self.x_unlabelled[i]
                additions_y[j] += corresponding_class[i]
                additions_cosine[j] += self.cosine_sim_matrix_unlabelled[i]
                self.shifted_samples[i].append(len(self.y_train)+j)
                j += 1

        # print(f"The old shape of the matrix is {self.x_train.shape}")
        self.x_train = np.concatenate((self.x_train, additions_x), axis=0)
        self.y_train = np.concatenate(
            (self.y_train, additions_y.reshape((-1,))), axis=0)

        self.cosine_sim_matrix_train = np.concatenate(
            (self.cosine_sim_matrix_train, additions_cosine), axis=0)
        # print(f"The new shape of the matrix is {self.x_train.shape}")
        print(additions_y)
        labels = self.le.inverse_transform(
            np.array(additions_y, dtype=np.int).reshape((-1,)))
        print(labels)
        print(
            f"The number of samples added to the training dataset is {sum(class_confidence)}")
        print()
        # add code for deleting the rows from the unlabelled dataset below.

        # cleaning up the unlabelled dataset before next iteration
        # self.__remove_entries_from_unlabelled__(class_confidence)
        return

    def ensemble_predictor(self):
        print("Entered the ensemble predictor")
        corresponding_class1 = self.model.predict(self.x_unlabelled)
        corresponding_class2 = self.knn.predict(self.x_unlabelled)
        corresponding_class3 = self.rfc.predict(self.x_unlabelled)
        corresponding_class4 = self.mnb.predict(self.x_unlabelled)
        print("finished predicting")

        valid_samples = np.zeros((len(corresponding_class1),))
        for i in range(len(corresponding_class1)):
            temp = set()
            temp.add(corresponding_class1[i])
            temp.add(corresponding_class2[i])
            temp.add(corresponding_class3[i])
            temp.add(corresponding_class4[i])
            if (len(temp) >= 2):
                valid_samples[i] = 1

        print(sum(valid_samples))
        return valid_samples

    def __determine_suitable_unlabelled_samples__(self):
        # determining suitable samples
        print("Determining suitable samples to add to the training dataset")
        corresponding_class = self.model.predict(self.x_unlabelled)
        valid_samples = self.ensemble_predictor()
        # We can first try a naive approach and where we predict the probabilities of each class for a given datapoint and if the probability of the class is high (greater than threshold then we can add that sample to the training set)
        predict_probs = self.model.predict_proba(self.x_unlabelled)

        class_confidence = predict_probs.max(axis=1)
        predicted_accuracy = np.copy(class_confidence)
        class_confidence[class_confidence >= self.threshold] = 1
        class_confidence[class_confidence != 1] = 0

        num_samples = 0
        for i in range(len(class_confidence)):
            if (class_confidence[i] and valid_samples[i]):  # and valid_samples[i]
                if (i in self.shifted_samples):
                    class_confidence[i] = 0
                    if (self.shifted_samples[i][0] != corresponding_class[i]):
                        # This time's prediction is different
                        if (self.shifted_samples[i][1] < predicted_accuracy[i]):
                            self.shifted_samples[i][1] = predicted_accuracy[i]
                            self.shifted_samples[i][0] = corresponding_class[i]
                            self.y_train[self.shifted_samples[i]
                                         [2]] = corresponding_class[i]
                else:

                    if (corresponding_class[i] == 3 or corresponding_class[i] == 10):
                        class_confidence[i] = 0
                        continue
                    num_samples += 1
                    if (num_samples >= round(self.max_added_samples*len(self.x_train))):
                        class_confidence[i:] = 0
                        break

                    self.shifted_samples[i] = [
                        corresponding_class[i], predicted_accuracy[i]]

            else:
                class_confidence[i] = 0

        # Now the array class_confidence contains 0s and 1s. All locations corresponding to 1s should be added to the training data set.
        return class_confidence, corresponding_class

    def write_pred_kaggle_file(self, outfname):
        """Writes the predictions in Kaggle format.

        Given the unlabeled object, classifier, outputfilename, and the speech object,
        this function write the predictions of the classifier on the unlabeled data and
        writes it to the outputfilename. The speech object is required to ensure
        consistent label names.
        """
        yp = self.model.predict(self.x_unlabelled)
        labels = self.le.inverse_transform(
            np.array(yp, dtype=np.int).reshape((-1,)))
        f = open(outfname, 'w')
        f.write("FileIndex,Category\n")
        for i in range(len(self.unlabeled.fnames)):
            fname = self.unlabeled.fnames[i]
            # iid = file_to_id(fname)
            f.write(str(i+1))
            f.write(",")
            # f.write(fname)
            # f.write(",")
            f.write(labels[i])
            f.write("\n")
        f.close()
        return

    def __evaluate_model__(self, count):

        print("Started Evaluating the Classifier")

        yp = self.model.predict(self.x_train)
        acc = metrics.accuracy_score(self.y_train, yp)
        self.train_acc_lst.append(100*acc)
        print("LR Training Accuracy :", acc)

        yp = self.model.predict(self.x_val)
        acc = metrics.accuracy_score(self.y_val, yp)
        self.val_acc_lst.append(100*acc)
        print("LR Testing Accuracy :", acc)
        print()

        self.write_pred_kaggle_file(f"./predFiles/speech-pred{count}.csv")
        ###############

        yp = self.rfc.predict(self.x_train)
        acc = metrics.accuracy_score(self.y_train, yp)
        print("RFC Training Accuracy :", acc)

        yp = self.rfc.predict(self.x_val)
        acc = metrics.accuracy_score(self.y_val, yp)
        print("RFC Testing Accuracy :", acc)
        print()
        ###############
        yp = self.mnb.predict(self.x_train)
        acc = metrics.accuracy_score(self.y_train, yp)
        print("mnb Training Accuracy :", acc)

        yp = self.mnb.predict(self.x_val)
        acc = metrics.accuracy_score(self.y_val, yp)
        print("mnb Testing Accuracy :", acc)
        print()
        ###############
        yp = self.knn.predict(self.x_train)
        acc = metrics.accuracy_score(self.y_train, yp)
        print("knn Training Accuracy :", acc)

        yp = self.knn.predict(self.x_val)
        acc = metrics.accuracy_score(self.y_val, yp)
        print("knn Testing Accuracy :", acc)

        print()
        return

    def display_all_documents_similarity(self):
        stop_words = set(stopwords.words("english"))
        self.txt_files = os.listdir("./allSpeeches")
        documents = [open(f"./allSpeeches/{speech}").read()
                     for speech in self.txt_files]
        self.tf_idf_vectorizer = TfidfVectorizer(
            stop_words=stop_words, preprocessor=custom_preprocessor, ngram_range=(1, 2))
        self.tfidf_documents = self.tf_idf_vectorizer.fit_transform(documents)

        # print(tfidf_documents.shape)
        pairwise_similarity = self.tfidf_documents * self.tfidf_documents.T
        # print(pairwise_similarity.toarray().shape)
        return

    def compute_cosine_similarity_based_accuracy(y, y_pred):
        acc = metrics.accuracy_score(y, y_pred)
        print("Accuracy :", acc)
        return

    def compute_cosine_similarity_matrix(self, X):
        # print(txt_files[0])
        y_pred = []
        # print(speech.dev_labels[0])
        cosine_sim_matrix = np.zeros((len(X), 19))
        for j in range(len(X)):

            doc_vec = self.tf_idf_vectorizer.transform([X[j]]).toarray()

            max_cosine_sim = 0
            pred_label = ''
            for i in range(self.tfidf_documents.shape[0]):

                cosine_sim = np.dot(
                    doc_vec, self.tfidf_documents[i].toarray().reshape((-1, 1)))[0][0]

                cosine_sim_matrix[j][i] = cosine_sim
                if (cosine_sim > max_cosine_sim):
                    max_cosine_sim = cosine_sim
                    pred_label = self.txt_files[i][:-4]

                # print(f"The cosine similarity is {cosine_sim}")
                # print()
            y_pred.append(pred_label)

        return cosine_sim_matrix, y_pred

    def dimensionality_reduction(self):
        pca = PCA(n_components=4370)

        trainX = pca.fit_transform(self.x_train)
        devX = pca.transform(self.x_val)
        unlabelledX = pca.transform(self.x_unlabelled)

        return trainX, devX, unlabelledX

    def train(self):
        i = 0
        self.display_all_documents_similarity()
        # Before we begin the training we should call the cosine_similarity measure to generate the modified training_data
        # kmeans = KMeans(n_clusters=19, random_state=0)
        self.cosine_sim_matrix_train, _ = self.compute_cosine_similarity_matrix(
            self.labelled_train)

        self.cosine_sim_matrix_val, _ = self.compute_cosine_similarity_matrix(
            self.labelled_dev)

        self.cosine_sim_matrix_unlabelled, y_cosine_pred = self.compute_cosine_similarity_matrix(
            self.unlabelled_text)

        # x_train, x_val, x_unlabelled = self.dimensionality_reduction()

        self.x_train = modify_data(self.x_train, self.cosine_sim_matrix_train)
        self.x_val = modify_data(self.x_val, self.cosine_sim_matrix_val)
        self.x_unlabelled = modify_data(
            self.x_unlabelled, self.cosine_sim_matrix_unlabelled)

        while (i < self.max_epochs and not self.is_completed):
            print(f"EPOCH: {i}")
            self.model.fit(self.x_train, self.y_train)
            # print("a")
            self.knn.fit(self.x_train, self.y_train)
            # print("b")
            self.mnb.fit(self.x_train, self.y_train)
            # print("c")
            self.rfc.fit(self.x_train, self.y_train)
            # print("d")

            self.__evaluate_model__(i)

            corresponding_class = self.model.predict(self.x_unlabelled)

            class_confidence, corresponding_class = self.__determine_suitable_unlabelled_samples__()
            self.__update_training_data__(
                class_confidence, corresponding_class)
            # now we do classify.predict on the unlabelled dataset and with these predictions update the training dataset

            i += 1
        # print(self.train_acc_lst)
        # print(self.val_acc_lst)
        self.plot_acc_trends()
        return


if __name__ == "__main__":
    pass
