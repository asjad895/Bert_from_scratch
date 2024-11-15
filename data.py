import re

text = (
       'Hello, how are you? I am Romeo.n'
       'Hello, Romeo My name is Juliet. Nice to meet you.n'
       'Nice meet you too. How are you today?n'
       'Great. My baseball team won the competition.n'
       'Oh Congratulations, Julietn'
       'Thanks you Romeo'
   )

def preprocess(data):
    sentences = re.sub("[.,!?-]", '', data.lower()).split('n')  # filter '.', ',', '?', '!'
    word_list = list(set(" ".join(sentences).split()))

    return word_list


class Data:
    def __init__(self,word_list):
        self.word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
        self.vocab = word_list

    
    def get_vocab(self):
        for i, w in enumerate(self.vocab):
            self.word_dict[w] = i + 4
        self.enoced_dict= {i: w for i, w in enumerate(self.word_dict)}
        vocab_size = len(self.word_dict)
        print(f"Vocab Size = {vocab_size}")