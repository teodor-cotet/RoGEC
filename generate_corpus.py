
"""restrictions for input files:
    -a sentence should not be splitted into multiple lines
"""

from os import listdir
from os.path import isdir, isfile, join
from nltk.tokenize import sent_tokenize, WordPunctTokenizer
from typing import List, Iterable
from gensim.models import FastText
import sys
import csv
sys.path.insert(0, '/home/teo/projects/Readerbench-python/')

from rb.parser.spacy_parser import SpacyParser
from rb.core.lang import Lang
from rb.core.document import Document
import random
import string
from typodistance import typoGenerator
from mistake import Mistake 


class CorpusGenerator():


    PATH_RAW_CORPUS = "books/"
    MIN_SENT_TOKENS = 4
    CORRECT_DIACS = {
        "ş": "ș",
        "Ş": "Ș",
        "ţ": "ț",
        "Ţ": "Ț",
    }
    MAX_SENT_TOKENS = 18
    FAST_TEXT = "/home/teo/projects/readme-models/models/fasttext2/fast_text.model"

    def __init__(self):
        self.files = []
        for f in listdir(CorpusGenerator.PATH_RAW_CORPUS):
            if isfile(join(CorpusGenerator.PATH_RAW_CORPUS, f)) and f.endswith(".txt"):
                self.files.append(join(CorpusGenerator.PATH_RAW_CORPUS, f))
        
        self.parser = SpacyParser.get_instance().get_model(Lang.RO)
        self.fasttext = FastText.load(CorpusGenerator.FAST_TEXT)

                
    def split_sentences(self, fileName: str) -> Iterable[List[str]]:
        # sentences = []
        tokenizer = WordPunctTokenizer()
        with open(fileName, "rt", encoding='utf-8') as f:
            for line in f.readlines():
                for sent in sent_tokenize(line):
                    yield sent

    def modify_word(self, token, mistke_type=Mistake.TYPO):
        text_token = token.text

        if mistke_type is Mistake.TYPO:
            typo = random.choice([x for x in typoGenerator(text_token, 2)][1:])
            return typo
        elif mistke_type is Mistake.INFLECTED:
            candidates = self.fasttext.similar_by_vector(token, topn=200)
            # check lemma for each candidate
            for fast_token, coss in candidates:
                if fast_token.

    
    def clean_text(self, text: str):
        list_text = list(text)
        # some cleaning correct diacritics + eliminate \
        text = "".join([CorpusGenerator.CORRECT_DIACS[c] if c in CorpusGenerator.CORRECT_DIACS else c for c in list_text])
        return text

    def generate(self):
        lines = []
        for ffile in self.files[:1]:
            for i, sent in enumerate(self.split_sentences(ffile)):
                line = []
                print(len(lines))
                if len(lines) > 12000:
                    break
                sent = self.clean_text(sent)
                docs_ro = self.parser(sent)
                tokens = [token for token in docs_ro]
                text_tokens = [token.text for token in docs_ro]
                sent2 = " ".join(text_tokens)
                
                if len(tokens) >= CorpusGenerator.MIN_SENT_TOKENS and len(tokens) <= CorpusGenerator.MAX_SENT_TOKENS:
                    count_tries = 0
                    while count_tries < 32:
                        index = random.randint(0, len(tokens) - 1)
                        tagg = str(tokens[index].tag_)
                        if (tagg.startswith("P") or tagg.startswith("COMMA")
                            or tagg.startswith("DASH") or tagg.startswith("COLON")): # punctuation
                            count_tries += 1
                            continue
                        try:
                            text_tok = self.modify_word(tokens[index], mistake_type=Mistake.INFLECTED)
                            text_tokens[index] = text_tok
                            sent3 = " ".join(text_tokens)
                            line.append(sent2)
                            line.append(sent3)
                            line.append(tokens[index].tag_)
                            line.append(Mistake.INFLECTED.value)
                            break
                        except:
                            count_tries += 1
                  
                if len(line) > 1:
                    lines.append(line)
        print(len(lines))
        with open('typos.csv', 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(lines)  
        # print(self.files)
        # for x in self.split_sentences(self.files[0]):
        #     print(x)

if __name__ == "__main__":
    corpusGenerator = CorpusGenerator()
    corpusGenerator.generate()
    