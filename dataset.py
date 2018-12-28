import re
from torchtext import data
import jieba

regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')


def word_cut(text):
    text = regex.sub(' ', text)
    return ' '.join([word for word in jieba.cut(text) if word.strip()]).split()


def get_dataset(path, text_field, label_field):
    text_field.tokenize = word_cut
    train, dev = data.TabularDataset.splits(
        path=path, format='tsv', skip_header=True,
        train='train.tsv', validation='dev.tsv',
        fields=[
            ('index', None),
            ('label', label_field),
            ('text', text_field)
        ]
    )
    return train, dev
