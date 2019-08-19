
"""restrictions for input files:
    -a sentence should not be splitted into multiple lines
"""

from os import listdir
from os.path import isdir, isfile, join
from nltk.tokenize import sent_tokenize, WordPunctTokenizer
from typing import List, Iterable
from gensim.models import FastText
from gensim.models.wrappers import FastText as FastTextWrapper
import sys
import csv
import argparse



import random
import string
from typodistance import typoGenerator
from mistake import Mistake 
"""
    TODO define categories of mistakes
"""
log = open("log.log", "w", encoding='utf-8')

class CorpusGenerator():

    PATH_RAW_CORPUS = "corpora/good/"
    MIN_SENT_TOKENS = 6
    CORRECT_DIACS = {
        "ş": "ș",
        "Ş": "Ș",
        "ţ": "ț",
        "Ţ": "Ț",
    }
    MAX_SENT_TOKENS = 18
    #FAST_TEXT = "/home/teo/projects/readme-models/models/fasttext_fb/wiki.ro"
    #FAST_TEXT = "/home/teo/repos/langcorrections/fasttext_fb/wiki.ro"
    GENERATE = 1e6
    TASK = Mistake.INFLECTED

    """ get files """
    def __init__(self):
        global args
        if args.task == "i":
            CorpusGenerator.TASK = Mistake.INFLECTED
        elif args.task == 't':
            CorpusGenerator.TASK = Mistake.TYPO
        else:
            CorpusGenerator.TASK = Mistake.ALL
        self.files = []
        for f in listdir(CorpusGenerator.PATH_RAW_CORPUS):
            if isfile(join(CorpusGenerator.PATH_RAW_CORPUS, f)) and f.endswith(".txt"):
                self.files.append(join(CorpusGenerator.PATH_RAW_CORPUS, f))
        
        #self.parser = SpacyParser.get_instance().get_model(Lang.RO)
        #self.fasttext = FastText.load(CorpusGenerator.FAST_TEXT)
        #self.fasttext = FastTextWrapper.load_fasttext_format(CorpusGenerator.FAST_TEXT)

    def test_parser(self):
        txt = """S-a născut aldkl repede la 1 februarie 1852,[3] 
                în satul Haimanale (care astăzi îi poartă numele),  1900
                fiind primul născut al lui Luca Ștefan Caragiale și al Ecaterinei Chiriac Karaboas.
                Conform unor surse, familia sa ar fi fost de origine aromână.[6] Tatăl său, Luca (1812 - 1870), 
                și frații acestuia, Costache și Iorgu, s-au născut la Constantinopol, 
                fiind fiii lui Ștefan, un bucătar angajat la sfârșitul anului 1812 de Ioan Vodă Caragea în suita sa."""

        self.construct_lemma_dict()
        docs_ro = Document(Lang.RO, txt)
        
        for key in docs_ro.get_tokens():
            print(key.text, key.lemma, key.pos, key.ent_type, key.detailed_pos, key.is_dict_word(), file=log)

            if key.lemma in self.lemma_to_words:
                print(self.lemma_to_words[key.lemma], file=log)

                for alternative in self.lemma_to_words[key.lemma]:
                    token = Document(Lang.RO, alternative).get_tokens()[0]
                    print(token.text, token.detailed_pos, token.is_dict_word(), file=log)
            else:
                print('-', file=log)
            print('\n\n', file=log)

    def split_sentences(self, fileName: str) -> Iterable[List[str]]:
        # sentences = []
        tokenizer = WordPunctTokenizer()
        with open(fileName, "rt", encoding='utf-8') as f:
            for line in f.readlines():
                for sent in sent_tokenize(line):
                    yield sent

    def modify_word(self, token, mistake_type):
        text_token = token.text
        try:
            if mistake_type is Mistake.TYPO:
                typo = random.choice([x for x in typoGenerator(text_token, 2)][1:])
                return typo
            elif mistake_type is Mistake.INFLECTED:
                if token.text not in self.word_to_lemma[token.text]:
                    lemma = token.lemma_
                else:
                    lemma = self.word_to_lemma[token.text]

                if lemma in self.lemma_to_words:
                    candidates = self.lemma_to_words[lemma]
                    potential = random.choice(candidates)
                    if potential != token.text:
                        return potential
            return None
        except:
            return None

    """ clean diacritics """
    def clean_text(self, text: str):
        list_text = list(text)
        text = "".join([CorpusGenerator.CORRECT_DIACS[c] if c in CorpusGenerator.CORRECT_DIACS else c for c in list_text])
        return text

    def construct_connectors(self, connectors_file="wordlists/connectives_ro"):
        pass
        
    def construct_lemma_dict(self, lemma_file="wordlists/lemmas_ro.txt"):

        self.word_to_lemma = {}
        self.lemma_to_words = {}

        with open(lemma_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = self.clean_text(line)
                line = line.split()
                if len(line) != 2:
                    continue

                line[0] = self.clean_text(line[0].strip())
                line[1] = self.clean_text(line[1].strip())
               
                self.word_to_lemma[line[0]] = line[1]
                self.word_to_lemma[line[1]] = line[1]

                if line[1] not in self.lemma_to_words:
                    self.lemma_to_words[line[1]] = [line[0]]
                else:
                    self.lemma_to_words[line[1]].append(line[0])
                
                if line[1] not in self.lemma_to_words:
                    self.lemma_to_words[line[1]] = [line[1]]
                else:
                    self.lemma_to_words[line[1]].append(line[1])

        """ some words are repeated """            
        for key, v in self.lemma_to_words.items():
            self.lemma_to_words[key] = list(set(v))

        print('lemmas: {}'.format(len(self.lemma_to_words)))
        print('words: {}'.format(len(self.word_to_lemma)))

    def construct_dict(self, dict_file="wordlists/dict_ro.txt"):
         self.dict = set()
         with open(dict_file, 'r') as f:
               for line in f:
                   line = line.strip()
                   self.dict.add(line)

    def is_not_word(self, tagg):
        return (tagg.startswith("P") or tagg.startswith("COMMA")
                                or tagg.startswith("DASH") or tagg.startswith("COLON")
                                or tagg.startswith("QUEST") or tagg.startswith("HELLIP")
                                or tagg.startswith("DBLQ") or tagg.startswith("EXCL")
                                or tagg.startswith("X"))

    def not_good_for_infl(self, tagg):
        return (tagg.startswith("P") or tagg.startswith("COMMA")
                                or tagg.startswith("DASH") or tagg.startswith("COLON")
                                or tagg.startswith("QUEST") or tagg.startswith("HELLIP")
                                or tagg.startswith("DBLQ") or tagg.startswith("EXCL")
                                or tagg.startswith("X") or tagg.startswith("Csssp")
                                or tagg.startswith("Qz")or tagg.startswith("Rp"))
    
    def generate(self):
        global args
        self.construct_lemma_dict()
        sentences = []

        for ffile in self.files:
            
            if len(sentences) > args.generate:
                break

            for _, sent in enumerate(self.split_sentences(ffile)):
                line = []
                if len(sentences) % 1000 == 0:
                    print('Generated {} samples'.format(len(sentences)))

                if len(sentences) > args.generate:
                    break

                sent = self.clean_text(sent)
                docs_ro = self.parser(sent)
                tokens = [token for token in docs_ro]

                # nr of words out of dictionary
                cnt_out_of_dict = 0
                for token in tokens:
                    tagg = str(token.tag_)
                    if  self.is_not_word(tagg) == False and token.text not in self.fasttext.wv.vocab:
                        cnt_out_of_dict += 1
                        #print(token.text)

                text_tokens = [token.text for token in docs_ro]
                sent2 = " ".join(text_tokens)

                try:
                    if (len(tokens) >= CorpusGenerator.MIN_SENT_TOKENS and len(tokens) <= CorpusGenerator.MAX_SENT_TOKENS
                    and cnt_out_of_dict == 0):
                        count_tries = 0
                        while count_tries < 20:
                            index = random.randint(0, len(tokens) - 1)
                            tagg = str(tokens[index].tag_)
                            if CorpusGenerator.TASK is Mistake.INFLECTED and self.not_good_for_infl(tagg) == True: # punctuation
                                count_tries += 1
                                continue

                            text_tok = self.modify_word(tokens[index], CorpusGenerator.TASK)
                            if text_tok is not None:
                                text_tokens[index] = text_tok
                                sent3 = " ".join(text_tokens)
                                line.append(sent2)
                                line.append(sent3)
                                line.append(tokens[index].tag_)
                                line.append(Mistake.INFLECTED.value)
                                break
                            else:
                                count_tries += 1
                                continue
                            count_tries += 1
                except:
                    line = []
                  
                if len(line) > 1:
                    sentences.append(line)
        with open(args.output, 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(sentences)  
        # print(self.files)
        # for x in self.split_sentences(self.files[0]):
        #     print(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--output', dest='output', action='store', default="out.csv")
    parser.add_argument('--task', dest='task', action='store', default="i")
    parser.add_argument('--rb_path', dest='rb_path', action='store', default="../readerbenchpy/")
    parser.add_argument('--generate', dest='generate', action='store', default=3e5, type=int)
    args = parser.parse_args()

    for k in args.__dict__:
        if args.__dict__[k] is not None:
            print(k, '->', args.__dict__[k])
    sys.path.insert(0, args.rb_path)

    from rb.parser.spacy_parser import SpacyParser
    from rb.core.lang import Lang
    from rb.core.document import Document
    from rb.core.pos import POS
    corpusGenerator = CorpusGenerator()
    #corpusGenerator.generate()
    corpusGenerator.test_parser()
