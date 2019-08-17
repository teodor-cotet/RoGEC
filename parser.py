import argparse


class Parser():

    def __init__(self):
        pass
    
    def parse_lang_8(self, lang8_train='lang-8-en-1.0/entries.train', lang8_test='lang-8-en-1.0/entries.test'):
        global args
        train, test = [], []

        with open(lang8_train, 'r', encoding='utf-8') as f:
            for line in f:
                cols = line.split('\t')
                if len(cols) == 6:
                    train.append((cols[4], cols[5]))
        
        with open(lang8_test, 'r', encoding='utf-8') as f:
            for line in f:
                cols = line.split('\t')
                if len(cols) == 6:
                    test.append((cols[4], cols[5]))

        return train, test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--lang8_train', dest='lang8_train', action='store', default='lang-8-en-1.0/entries.train')
    parser.add_argument('--lang8_test', dest='lang8_test', action='store', default='lang-8-en-1.0/entries.test')
    
    args = parser.parse_args()

    for k in args.__dict__:
        if args.__dict__[k] is not None:
            print(k, '->', args.__dict__[k])
    
    parser = Parser()
    train, test = parser.parse_lang_8()
