A series of standard Machine Learning (classification) algorithms implemented in Java. The repository includes:

- AdaBoost
- Decision Trees
- Decision Forests
- kNN
- Perceptron Learners
- Neural Networks

Each of these implementations satisfy the Classifier interface.

Additionally, BaselineClassifier.java provides a classifier that simply predicts every example to be the most frequently occuring classification in the data set, as a point of comparison for the other implementations.

## Data Sets

DataSet.java allows users to read data from a data set as specified by the 'example' files in the data/ directory. Once parsed, users can easily extract information about the target data set using the DataSet class.

TestHarness.java can be used to test any Classifier on any DataSet through the use of a holdout set.

## Contact

For any questions, please reach out to David Dohan (ddohan@princeton.edu) or Charlie Marsh (crmarsh@princeton.edu).
