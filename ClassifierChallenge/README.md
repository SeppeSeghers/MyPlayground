## README

The purpose of this notebook, is to compare and explore different models to predict classified data.
In this code, there are 5 main methods covered
1. KNN
    - Different k-values
2. LogReg
    - Uniform weights
    - Balanced weights
3. Decision Tree
    - Different depths
4. SVM
    - Different c-values
    - Various kernel types
5. Naive Bayes


All models are applied on [this dataset](https://www.kaggle.com/uciml/mushroom-classification).

The data is preprocessed using LabelEncoder and StandardScaler, and split randomly into a 3/4 training set, and a 1/4 testing set to asses the accuraccy of each model.
All accuracies are computed via metrics.accuracy_score()