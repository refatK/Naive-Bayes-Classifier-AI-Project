from codecs import open
import random
import json


# Used for Task 0
def read_docs(doc_file='data.txt'):
    with open(doc_file, encoding='utf-8') as f:
        data_list = []
        for line in f:
            word_list = line.strip().split()

            data_list.append({
                'label': word_list[1],
                'doc': word_list[3:]
            })
    return data_list


def split_data(data: list, train_percent=0.8, randomize=False, seed: int = None):
    if randomize:
        if seed is None:
            random.shuffle(data)
        else:
            random.Random(seed).shuffle(data)

    split_point = int(train_percent * len(data))
    train_set = data[:split_point]
    test_set = data[split_point:]
    return train_set, test_set


# -----


# Used for Task 3
def accuracy(true_labels, guessed_labels):
    total = len(true_labels)
    num_correct = sum(1 for true, guess in zip(true_labels, guessed_labels) if true == guess)
    return num_correct / total


# Used for Task 4
def detailed_info(true_labels, guessed_labels, docs):
    detail_string = ''
    correct_string = ''
    false_pos_string = ''
    false_neg_string = ''

    total = len(true_labels)

    num_correct = 0
    num_wrong = 0
    num_false_pos = 0
    num_false_neg = 0
    num_true_pos = 0
    num_true_neg = 0

    for doc, true, guess in zip(docs, true_labels, guessed_labels):
        if true == guess:
            num_correct += 1
            correct_string += f'    "{" ".join(doc)}"\n'
            
        if guess == 'pos' and true == 'pos':
            num_true_pos += 1

        if guess == 'neg' and true == 'neg':
            num_true_neg += 1

        if guess == 'pos' and true == 'neg':
            num_false_pos += 1
            false_pos_string += f'    "{" ".join(doc)}"\n'

        if guess == 'neg' and true == 'pos':
            num_false_neg += 1
            false_neg_string += f'    "{" ".join(doc)}"\n'

    num_wrong = num_false_pos + num_false_neg
    acc = num_correct / total

    detail_string += f'''
Classification Results:

Correct: {num_correct}
Wrong: {num_wrong}
  False Positives: {num_false_pos}
  False Negatives: {num_false_neg}
Accuracy: {acc}

Correct Docs:
{correct_string}

False Positive Docs:
{false_pos_string}

False Negative Docs:
{false_neg_string}

    '''

    return {
        'correct': num_correct,
        'wrong': num_wrong,
        'false_pos': num_false_pos,
        'false_neg': num_false_neg,
        'true_pos': num_true_pos,
        'true_neg': num_true_neg,
        'detail_string': detail_string
    }
