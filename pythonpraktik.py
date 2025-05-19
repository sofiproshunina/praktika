import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

nltk.download('stopwords')

# Загрузка данных
df_neg = pd.read_csv('negative.csv', header=None, sep=';', encoding='utf-8', engine='python', on_bad_lines='skip')
df_pos = pd.read_csv('positive.csv', header=None, sep=';', encoding='utf-8', engine='python', on_bad_lines='skip')
df_neg['label'] = 0
df_pos['label'] = 1
df = pd.concat([df_neg, df_pos], ignore_index=True)
df.columns = ['id', 'user_id', 'username', 'text', 
              'col5', 'col6', 'col7', 'col8', 
              'col9', 'col10', 'col11', 'col12', 'label']
russian_stopwords = set(stopwords.words('russian'))
morph = MorphAnalyzer()

#Лемматизация текста
lemma_cache = {}
def lemmatize_word(word):
    if word not in lemma_cache:
        lemma_cache[word] = morph.parse(word)[0].normal_form
    return lemma_cache[word]

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|@\w+|#\w+|[^а-яА-ЯёЁ\s]', ' ', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in russian_stopwords]
    lemmas = [lemmatize_word(word) for word in tokens]
    return ' '.join(lemmas)

print("Начинаю лемматизацию...")
df['clean_text'] = df['text'].apply(preprocess)
print("Лемматизация завершена!")

#Пример заполнения
print(df[['clean_text', 'label']].head())

# Разделение данных
X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'\nРазмер обучающей выборки: {X_train.shape[0]}')
print(f'Размер тестовой выборки: {X_test.shape[0]}')

# Векторизация с CountVectorizer
# Униграммы
vectorizer_uni = CountVectorizer(ngram_range=(1,1))
X_train_uni = vectorizer_uni.fit_transform(X_train)
X_test_uni = vectorizer_uni.transform(X_test)
print(f'\nУниграммы: Размер обучающей выборки {X_train_uni.shape}, тестовой {X_test_uni.shape}')
# Биграммы
vectorizer_bi = CountVectorizer(ngram_range=(2,2))
X_train_bi = vectorizer_bi.fit_transform(X_train)
X_test_bi = vectorizer_bi.transform(X_test)
print(f'Биграммы: Размер обучающей выборки {X_train_bi.shape}, тестовой {X_test_bi.shape}')
# Триграммы
vectorizer_tri = CountVectorizer(ngram_range=(3,3))
X_train_tri = vectorizer_tri.fit_transform(X_train)
X_test_tri = vectorizer_tri.transform(X_test)
print(f'Триграммы: Размер обучающей выборки {X_train_tri.shape}, тестовой {X_test_tri.shape}')


# Векторизация с TfidfVectorizer
# Униграммы
tfidf_uni = TfidfVectorizer(ngram_range=(1,1))
X_train_tfidf_uni = tfidf_uni.fit_transform(X_train)
X_test_tfidf_uni = tfidf_uni.transform(X_test)
# Биграммы
tfidf_bi = TfidfVectorizer(ngram_range=(2,2))
X_train_tfidf_bi = tfidf_bi.fit_transform(X_train)
X_test_tfidf_bi = tfidf_bi.transform(X_test)
# Триграммы
tfidf_tri = TfidfVectorizer(ngram_range=(3,3))
X_train_tfidf_tri = tfidf_tri.fit_transform(X_train)
X_test_tfidf_tri = tfidf_tri.transform(X_test)


def train_and_evaluate(X_train_vec, X_test_vec, y_train, y_test, ngram_type):
    print(f"\nОбучение модели на {ngram_type}...")
    model = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Точность на тестовой выборке ({ngram_type}): {acc:.4f}")
    
    report = classification_report(y_test, y_pred, digits=4)
    print(f"Classification report ({ngram_type}):\n{report}")
    
    f1_macro = f1_score(y_test, y_pred, average='macro')
    print(f"Macro-average F1-score ({ngram_type}): {f1_macro:.4f}")
    
    return model

print("\n\nМодели с CountVectorizer")
model_unigram = train_and_evaluate(X_train_uni, X_test_uni, y_train, y_test, "униграммах CountVectorizer")
model_bigram = train_and_evaluate(X_train_bi, X_test_bi, y_train, y_test, "биграммах CountVectorizer")
model_trigram = train_and_evaluate(X_train_tri, X_test_tri, y_train, y_test, "триграммах CountVectorizer")

print("\n\nМодели с TfidfVectorizer")
model_tfidf_uni = train_and_evaluate(X_train_tfidf_uni, X_test_tfidf_uni, y_train, y_test, "униграммах TF-IDF")
model_tfidf_bi = train_and_evaluate(X_train_tfidf_bi, X_test_tfidf_bi, y_train, y_test, "биграммах TF-IDF")
model_tfidf_tri = train_and_evaluate(X_train_tfidf_tri, X_test_tfidf_tri, y_train, y_test, "триграммах TF-IDF")
