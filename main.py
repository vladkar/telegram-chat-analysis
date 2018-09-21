from html.parser import HTMLParser
from bs4 import BeautifulSoup
import glob, os
import pandas as pd
import string
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import RussianStemmer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


def parse(dir):
    files = []
    for file in glob.glob(f'{dir}/messages*'):
        files.append(file)

    fid = 0
    messages = []
    for file in files:
        fid += 1
        print('parsing', fid, len(files))
        doc = BeautifulSoup(open(file), 'html.parser')
        doc_messages = doc.find_all('div', ['message default clearfix', 'message default clearfix joined'])
        messages.extend(doc_messages)

    data = {}

    id = 0
    for raw_message in messages:
        id += 1
        if id % 100 == 0:
            print('processing', id, len(messages))
        author = raw_message.find('div', class_='initials')
        author_name = raw_message.find('div', class_='from_name')
        if author is not None:
            last_author = author
            last_author_name = author_name

        message = raw_message.find('div', class_='text')
        date = raw_message.find('div', class_='pull_right date details')

        if message is not None:
            author_data = last_author.text.strip()
            author_name_data = last_author_name.text.strip()
            timestamp_data = pd.to_datetime(date['title'], dayfirst=True)
            text_data = message.text.strip()

            data[id] = (author_data, author_name_data, timestamp_data, text_data)

    df = pd.DataFrame.from_dict(data, orient='index', columns=['author_initials', 'author_name', 'timestamp', 'text'])

    df.to_csv('crab_data.csv', encoding='utf-8')


def plot_general_activity():
    df = pd.read_csv('crab_data.csv').drop('Unnamed: 0', axis=1)
    df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index)

    df_day = df.resample('M', how={'score': 'count'})


    plt.plot(df_day.index, df_day[('score', 'text')])
    plt.title('Крабовая активность')
    plt.xlabel('Время')
    plt.ylabel('Сообщений в месяц')
    plt.show()


def process(user=None):
    df = pd.read_csv('crab_data.csv').drop('Unnamed: 0', axis=1)
    df = df.set_index('timestamp')
    df.index = pd.to_datetime(df.index)
    # nltk.download('stopwords')

    punc = set(string.punctuation)
    punc_add = {u'?'}
    punc = punc.union(punc_add)

    stop_nltk = set(stopwords.words('russian'))
    badwords = {
        u'я', u'а', u'да', u'но', u'тебе', u'мне', u'ты', u'и', u'у', u'на', u'ща', u'ага',
        u'так', u'там', u'какие', u'который', u'какая', u'туда', u'давай', u'короче', u'кажется', u'вообще',
        u'ну', u'не', u'чет', u'неа', u'свои', u'наше', u'хотя', u'такое', u'например', u'кароч', u'как-то',
        u'нам', u'хм', u'всем', u'нет', u'да', u'оно', u'своем', u'про', u'вы', u'м', u'тд',
        u'вся', u'кто-то', u'что-то', u'вам', u'это', u'эта', u'эти', u'этот', u'прям', u'либо', u'как', u'мы',
        u'просто', u'блин', u'очень', u'самые', u'твоем', u'ваша', u'кстати', u'вроде', u'типа', u'пока', u'ок',
        u'че', u'чо', u'чё', u'go'}
    stop = stop_nltk.union(badwords)

    stemmer = RussianStemmer()

    # Filter by author
    if user is not None:
        df = df[df['author_initials'] == user]

    # Filter punctuation
    df['text'] = df['text'].apply(lambda doc: ' '.join([word for word in doc.lower().split() if word not in punc]))

    # Filter service tags
    df['text'] = df['text'].apply(lambda doc: '' if 'server' in doc or 'getgif' in doc or 'if' in doc else doc)

    # Filter stop words
    df['text'] = df['text'].apply(lambda doc: ' '.join([word for word in doc.split() if word not in stop]))

    # Filter long words
    df['text'] = df['text'].apply(lambda doc: ' '.join([word for word in doc.split() if len(word) <= 20]))

    # Filter links http
    df['text'] = df['text'].apply(lambda doc: re.sub(r'http\S+', '', doc))

    # Filter links www
    df['text'] = df['text'].apply(lambda doc: re.sub(r'www\S+', '', doc))

    # Filter digits
    df['text'] = df['text'].apply(lambda doc: ''.join([i for i in doc if not i.isdigit()]))

    # Filter spaces
    df['text'] = df['text'].apply(lambda doc: ''.join([i for i in doc if i.isalpha() or i.isspace()]))

    # Stem
    df['text'] = df['text'].apply(lambda doc: ' '.join([stemmer.stem(word) for word in doc.lower().split()]))

    bag = {}
    for i, row in df.iterrows():
        for word in row['text'].split():
            if word not in bag:
                bag[word] = 0
            bag[word] += 1

    bag = sorted(bag.items(), key=lambda item: item[1], reverse=True)

    print(bag[:20])

    bag = bag[:20]

    plt.bar(range(len(bag)), [el[1] for el in bag])
    plt.xticks(range(len(bag)), [el[0] for el in bag], rotation='vertical')
    plt.title(f'Word frequency for {user}')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.show()


parse('ChatExport_07_09_2018')
process('DV')
