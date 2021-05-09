from reviewclassifierhelper import read_docs, split_data, accuracy, detailed_info
from naivebayesclassifier import NaiveBayesClassifier
import json

# Variable to modify before running the script
write_to_file = True
test_file = 'Sample2.txt'


# %% TASK 0: **Keep relevant data and split data**
# 0.1: only keep sentiment label and review text
review_data = read_docs(doc_file='data.txt')
# 0.2 get training and test set
review_train_set, review_test_set = split_data(review_data, train_percent=0.8, randomize=True, seed=None)

if write_to_file:
    with open('outputs/task0.txt', 'w+', encoding="utf-8") as file:
        file.write(f'TASK 0:\n\n')
        file.write(f'Training Set [has {len(review_train_set)}] docs\n')
        json.dump(review_train_set, file)
        file.write(f'\n\nTest Set [has {len(review_test_set)}] docs\n')
        json.dump(review_test_set, file)


# %% TASK 1: **Train the Naive Bayes Classifier**
train_docs = [data['doc'] for data in review_train_set]
train_labels = [data['label'] for data in review_train_set]
nb_classifier = NaiveBayesClassifier(smoothing_factor=2.6)
nb_classifier.train_classifier(train_docs, train_labels, vocab_from_train=True)

if write_to_file:
    with open('outputs/task1.txt', 'w+', encoding="utf-8") as file:
        file.write(f'TASK 1:\n\n')
        file.write(f'Used smoothing factor: {nb_classifier.smoothing_factor}\n\n')
        file.write(f'Generated Frequency Table:\n')
        json.dump(nb_classifier.freq_table, file)
        file.write(f'\n\nGenerated Probability Table:\n')
        json.dump(nb_classifier.p_table, file)


# %% TASK 2: **Classify documents and ensure classifier makes sense**
if write_to_file:
    with open('outputs/task2.txt', 'w+', encoding="utf-8") as file:
        file.write(f'TASK 2:\n\n')

        file.write(f'Sanity check:\n')

        sanity_docs = ['great'.split(), 'bad'.split(), 'a top-quality performance'.split()]
        for doc in sanity_docs:
            file.write(f'    Word = "{" ".join(doc)}"\n')
            file.write(f'        Pos score = {nb_classifier.score_doc_for_category(doc, "pos")}\n'
                       f'        Neg score = {nb_classifier.score_doc_for_category(doc, "neg")}\n'
                       f'        Guessed class = {nb_classifier.classify(doc)}\n\n')


# %% TASK 3: **Evaluate classifier**
test_docs = [data['doc'] for data in review_test_set]
test_labels = [data['label'] for data in review_test_set]

review_guessed_labels = nb_classifier.classify_all(test_docs)
acc = accuracy(test_labels, review_guessed_labels)
print(acc)

if write_to_file:
    with open('outputs/task3.txt', 'w+') as file:
        file.write(f'TASK 3:\n\n')
        file.write(f'Accuracy = {acc}')


# %% TASK 4: **Error Analysis**
detail = detailed_info(test_labels, review_guessed_labels, test_docs)

if write_to_file:
    with open('outputs/task4.txt', 'w+', encoding="utf-8") as file:
        file.write(f'TASK 4:\n\n')
        file.write(detail['detail_string'])


# %% DEMO (classifying the already trined clasifier on a set of review data given during a project demo)
new_data = read_docs(doc_file=test_file)

new_docs = [data['doc'] for data in new_data]
new_labels = [data['label'] for data in new_data]

new_guessed_labels = nb_classifier.classify_all(new_docs)
acc = accuracy(new_labels, new_guessed_labels)
print(f'Accuracy on {test_file} = {acc}')

new_detail = detailed_info(new_labels, new_guessed_labels, new_docs)

with open('outputs/demo.txt', 'w+', encoding="utf-8") as file:
    file.write(f'Demo Results:\n\n')
    file.write(new_detail['detail_string'])
