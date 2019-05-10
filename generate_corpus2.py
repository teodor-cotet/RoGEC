
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
sys.path.insert(0, '/home/teo/repos/Readerbench-python/')

from rb.parser.spacy_parser import SpacyParser
from rb.core.lang import Lang
from rb.core.document import Document
import random
import string
from typodistance import typoGenerator
from mistake import Mistake 


class CorpusGenerator():


    PATH_RAW_CORPUS = "books/"
    MIN_SENT_TOKENS = 6
    CORRECT_DIACS = {
        "ş": "ș",
        "Ş": "Ș",
        "ţ": "ț",
        "Ţ": "Ț",
    }
    MAX_SENT_TOKENS = 18
    #FAST_TEXT = "/home/teo/projects/readme-models/models/fasttext_fb/wiki.ro"
    FAST_TEXT = "/home/teo/repos/langcorrections/fasttext_fb/wiki.ro"

    def __init__(self):
        self.files = []
        for f in listdir(CorpusGenerator.PATH_RAW_CORPUS):
            if isfile(join(CorpusGenerator.PATH_RAW_CORPUS, f)) and f.endswith(".txt"):
                self.files.append(join(CorpusGenerator.PATH_RAW_CORPUS, f))
        
        self.parser = SpacyParser.get_instance().get_model(Lang.RO)
        #self.fasttext = FastText.load(CorpusGenerator.FAST_TEXT)
        self.fasttext = FastTextWrapper.load_fasttext_format(CorpusGenerator.FAST_TEXT)
                
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
                candidates = self.fasttext.similar_by_word(token.lemma_, topn=200)

                for fast_token, coss in candidates:
                    #print(self.parser(fast_token)[0].lemma_)
                    if self.parser(fast_token)[0].lemma_ == token.lemma_ and fast_token != token.text:
                        #print(fast_token, token)
                        return fast_token
            return None
        except:
            return None
    
    def clean_text(self, text: str):
        list_text = list(text)
        # some cleaning correct diacritics + eliminate \
        text = "".join([CorpusGenerator.CORRECT_DIACS[c] if c in CorpusGenerator.CORRECT_DIACS else c for c in list_text])
        return text.lower()

    def generate(self):
        lines = []
        for ffile in self.files:
            for i, sent in enumerate(self.split_sentences(ffile)):
                line = []
                print(len(lines))
                if len(lines) > 100000:
                    break
                sent = self.clean_text(sent)
                docs_ro = self.parser(sent)
                tokens = [token for token in docs_ro]
                text_tokens = [token.text for token in docs_ro]
                sent2 = " ".join(text_tokens)
                try:
                    if len(tokens) >= CorpusGenerator.MIN_SENT_TOKENS and len(tokens) <= CorpusGenerator.MAX_SENT_TOKENS:
                        count_tries = 0
                        while count_tries < 20:
                            index = random.randint(0, len(tokens) - 1)
                            tagg = str(tokens[index].tag_)
                            if (tagg.startswith("P") or tagg.startswith("COMMA")
                                or tagg.startswith("DASH") or tagg.startswith("COLON")): # punctuation
                                count_tries += 1
                                continue
                            text_tok = self.modify_word(tokens[index], Mistake.INFLECTED)
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
                    lines.append(line)
        print(len(lines))
        with open('inflected.csv', 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(lines)  
        # print(self.files)
        # for x in self.split_sentences(self.files[0]):
        #     print(x)

if __name__ == "__main__":
    corpusGenerator = CorpusGenerator()
    corpusGenerator.generate()
    
