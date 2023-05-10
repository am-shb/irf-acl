import numpy as np
import pandas as pd
import openai
from openai.embeddings_utils import get_embedding
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import tqdm
tqdm.tqdm.pandas()


openai.api_key = ""

GET_EMBEDDINGS = False

if GET_EMBEDDINGS:
    train_df = pd.read_csv('data/train_data.csv')
    val_df = pd.read_csv('data/val_data.csv')
    test_df = pd.read_csv('data/test_data.csv')


    train_df['embedding'] = train_df.text.progress_apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    train_df.to_csv('data/train_with_embeddings.csv')

    val_df['embedding'] = val_df.text.progress_apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    val_df.to_csv('data/val_with_embeddings.csv')

    test_df['embedding'] = test_df.text.progress_apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
    test_df.to_csv('data/test_with_embeddings.csv')

train_df = pd.read_csv('data/train_with_embeddings_pre.csv')
val_df = pd.read_csv('data/val_with_embeddings_pre.csv')
test_df = pd.read_csv('data/test_with_embeddings_pre.csv')

X_train = list(train_df.embedding.apply(eval).apply(np.array).values)
y_train_belong = train_df.belong.to_numpy()
y_train_burden = train_df.burden.to_numpy()

X_val = list(val_df.embedding.apply(eval).apply(np.array).values)
y_val_belong = val_df.belong.to_numpy()
y_val_burden = val_df.burden.to_numpy()

X_test = list(test_df.embedding.apply(eval).apply(np.array).values)
y_test_belong = test_df.belong.to_numpy()
y_test_burden = test_df.burden.to_numpy()

print('Loaded data with embeddings.')

belong_classifiers = {
    'lr': LogisticRegression(),
    'rf': RandomForestClassifier(),
    'svm': LinearSVC(),
    'mlp': MLPClassifier(learning_rate='adaptive', max_iter=100),
    'xgb': XGBClassifier(),
}
burden_classifiers = {
    'lr': LogisticRegression(),
    'rf': RandomForestClassifier(),
    'svm': LinearSVC(),
    'mlp': MLPClassifier(learning_rate='adaptive', max_iter=1000),
    'xgb': XGBClassifier(),
}

for (belong_clf_name, belong_clf), (burden_clf_name, burden_clf) in zip(belong_classifiers.items(), burden_classifiers.items()):
    belong_clf.fit(X_train, y_train_belong)
    belong_preds = belong_clf.predict(X_val)
    belong_precision, belong_recall, belong_f1, _ = precision_recall_fscore_support(y_val_belong, belong_preds, average='binary')
    belong_accuracy = accuracy_score(y_val_belong, belong_preds)
    
    burden_clf.fit(X_train, y_train_burden)
    burden_preds = burden_clf.predict(X_val)
    burden_precision, burden_recall, burden_f1, _ = precision_recall_fscore_support(y_val_burden, burden_preds, average='binary')
    burden_accuracy = accuracy_score(y_val_burden, burden_preds)

    avg_precision = (belong_precision + burden_precision) / 2
    avg_recall = (belong_recall + burden_recall) / 2
    avg_f1 = (belong_f1 + burden_f1) / 2
    avg_accuracy = (belong_accuracy + burden_accuracy) / 2

    print(f'=== {belong_clf_name} ===')
    print(f'belong_accuracy: {belong_accuracy}')
    print(f'burden_accuracy: {burden_accuracy}')
    print(f'avg_accuracy: {avg_accuracy}')
    print(f'belong_precision: {belong_precision}')
    print(f'burden_precision: {burden_precision}')
    print(f'avg_precision: {avg_precision}')
    print(f'belong_recall: {belong_recall}')
    print(f'burden_recall: {burden_recall}')
    print(f'avg_recall: {avg_recall}')
    print(f'belong_f1: {belong_f1}')
    print(f'burden_f1: {burden_f1}')
    print(f'avg_f1: {avg_f1}')
    print('\n')
