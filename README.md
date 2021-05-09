# Naive Bayes Classifier Project

- [Project Summary](#Project-Summary)
- [Running The Project](#Running-The-Project)
- [Implementation Details and Classifier Analysis](#Implementation-Details-and-Classifier-Analysis)

---

## Project Summary

This project involves using the "Naive Bayes Classification" method to take a list of labeled data involving reviews, and using the classification method to classify reviews as "positive" or "negative", using the individual words in the reviews and their labels to train a classifier on a set of that data (training set). The accuracy of the classifier was then tested on the other subset of the data that was not trained on (test set), by comparing the classified labels to the actual labels already given to the reviews.

The goal of the project was to manually implement this algorithm, and it does not use a machine learning library to do the Naive Bayes method.

Data Preperation/Cleanup was done to help the algorithm work at its best as well.

## Running The Project

1. Clone the repo locally using Git (or download the source files).
2. Open Command Line and nagivate to the project folder (folder that contains the file `nb.py`).
3. Run the script using `python nb.py` in Command Line.

### Note

- If you get missing module errors, there are two possible dependecies missing, run the following to have them all
  - `pip install numpy`
  - `pip install stop-words`
- You can use an IDE or Anaconda Prompt to run the script instead if preferred. The script that is used to run the project (`nb.py`) is also broken down as Jupyter Notebook cells, so each section can be ran and analyzed per step if needed.

## Implementation Details and Classifier Analysis

A PDF report was made for this project describing the difficulties implementing the classifier, the accuracy of the classifier, and analysis regarding why the accuracy might be the way it was, as well as approaches used to try and icrease the accuracy.

The report can be accessed [by clicking here](https://github.com/refatK/Naive-Bayes-Classifier-AI-Project/blob/main/Report%20and%20Analysis.pdf).
