
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
import copy
import re


import random
import string
from typodistance import typoGenerator
from mistake import Mistake 
from enum import Enum
from collections import Counter
"""
    TODO define categories of mistakes
"""
log = open("corpus_log.log", "w", encoding='utf-8')

class MistakeType(Enum):
    TYPO = 'typo' # typo
    DIAC = 'diac' # diacritice 
    Fv = 'vf' # forma verbului 
    SVA = 'sva' # subiect-verb
    PRE = 'pre' # preopozitii
    VIR = 'vir' # virgula
    FC = 'fc' # forma cuvintelor
    CR = 'cr' # cratima
    SP = 'sp' # spatiere
    GEN = 'gen' # genul gresit  
    CASE = 'case' # cazul substantivului
    PI = 'pi' # forma pronumelui de intarire
    ACP = 'acp' # acordul cu posesorul

class CorpusGenerator():

    PATH_RAW_CORPUS = "corpora/"
    MIN_SENT_TOKENS = 6
    CORRECT_DIACS = {
        "ş": "ș",
        "Ş": "Ș",
        "ţ": "ț",
        "Ţ": "Ț",
    }
    DIACS = {'Ă': 'A', 'ă': 'a', 'Â': 'A', 'â': 'a',
                    'Î':' I', 'î': 'i', 'Ș': 'S', 'ș': 's', 'Ț': 'T', 'ț': 't'}
    MAX_SENT_TOKENS = 18
    GENERATE = 1e6
    TASK = Mistake.INFLECTED

    """ get files """
    def __init__(self):
        global args
        self.files = []
        for f in listdir(CorpusGenerator.PATH_RAW_CORPUS):
            if isfile(join(CorpusGenerator.PATH_RAW_CORPUS, f)) and f.endswith(".txt"):
                self.files.append(join(CorpusGenerator.PATH_RAW_CORPUS, f))
                
    def split_sentences(self, fileName: str) -> Iterable[List[str]]:
        tokenizer = WordPunctTokenizer()
        with open(fileName, "rt", encoding='utf-8') as f:
            for line in f.readlines():
                for sent in sent_tokenize(line):
                    yield sent

    def clean_text(self, text: str):
        list_text = list(text)
        text = "".join([CorpusGenerator.CORRECT_DIACS[c] if c in CorpusGenerator.CORRECT_DIACS else c for c in list_text])
        return text

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

    def pre_computations(self):
        count = 0
        self.dash_words = {} # no-dash word: dash word
        for ffile in self.files:
            for i, correct_sent in enumerate(self.split_sentences(ffile)):
                docs_ro = Document(Lang.RO, correct_sent)
                tokens = docs_ro.get_tokens()
                if self.filter_sentence(correct_sent, tokens) == False: continue
                count += 1
                if count > args.samples: 
                    print(self.dash_words, file=log)
                    return
                for i, token in enumerate(tokens):
                    if token.text == '-' and i > 0 and i < len(tokens) - 1:
                        dash_word =  tokens[i - 1].text + tokens[i].text + tokens[i + 1].text
                        no_dash_word = tokens[i-1].text + tokens[i+1].text
                        self.dash_words[no_dash_word] = dash_word
                
    def is_wrongable_pos(self, pos):
        return (pos is not POS.X and pos is not POS.SYM and
                    pos is not POS.SPACE and pos is not POS.PUNCT)

    def filter_sentence(self, correct_sent, tokens):
        """filter wierd sentences """
        if not (correct_sent[0].isupper() or correct_sent[0] == '—'):  return False
        if correct_sent[0] == '-' and random.randint(0, 99) > 85:   return False
        if len(tokens) < 4: return False

        poses = [token.pos for token in tokens]
        pos_counter = Counter(poses)
        ok_words_to_modify = 0
        # for pos in POS: TODO include this
        #     if self.is_wrongable_pos(pos):
        #         ok_words_to_modify += pos_counter[pos]
        # if ok_words_to_modify == 0:
        #     return False
        return True

    def eliminate_one_diac(self, word_text):
        diacs = {'Ă': 'A', 'ă': 'a', 'Â': 'A', 'â': 'a',
                'Î':' I', 'î': 'i', 'Ș': 'S', 'ș': 's', 'Ț': 'T', 'ț': 't'}
        chars = list(word_text)
        chars_ind = [(i, c) for i, c in enumerate(chars) if c in diacs]
        (index, c) = random.choice(chars_ind)
        chars[index] = diacs[c]
        return "".join(chars)

    def modify_word(self, token, mistake_type):
        text_token = token.text
        try:
            # check pos
            if mistake_type is MistakeType.TYPO:
                typo = random.choice([x for x in typoGenerator(text_token, 1)][1:])
                return typo
            elif mistake_type is MistakeType.DIAC:
                if self.count_diacritics(text_token) >= 1:
                    return self.eliminate_one_diac(text_token)
                else:
                    return None
            elif mistake_type is MistakeType.Fv:
                return None
        except:
            return None
    
    def count_diacritics(self, s):
        diacs = {'Ă': 'A', 'ă': 'a', 'Â': 'A', 'â': 'a',
                'Î':' I', 'î': 'i', 'Ș': 'S', 'ș': 's', 'Ț': 'T', 'ț': 't'}
        count = 0 
        for c in s:
            if c in diacs:
                count += 1
        return count

    def replace_space_with_comma(self, s):
        lkk = list(s)
        lkk[0] = ','
        return "".join(lkk)
    
    def capitalize_first_char(self, s):
        lkk = list(s)
        lkk[1] = lkk[1].toupper()
        return "".join(lkk)

    def construct_prepositions_mistakes(self, correct_sent):

        success = False
        short_long = 'short-long'
        long_short = 'long-short'
        different = 'different'
        
        " types of replaces, first char has to be a space"
        replaces = {
            short_long: {" care ": " pe care ", ' ca ': ' ca și '},
            long_short: {" pe care ": " care ", ' ca și ': ' ca '},
            different: {' dintre ': ' din '}
        }

        for k, v in replaces.items():
            for kk, vv in replaces.items():
                nkk = self.replace_space_with_comma(kk)
                nvv = self.replace_space_with_comma(vv)
                replaces[k][nkk] = nvv
                
        wrong_sent = None
        randoms = [0, 1, 2]
        random.shuffle(randoms)
        for r in randoms:
            if r == 0:
                for short, llong in replaces[short_long].items():
                    if correct_sent.find(llong) == -1 and correct_sent.find(short) != -1:
                        wrong_sent = correct_sent.replace(short, llong)
                        return wrong_sent
            elif r == 1:
                for llong, short in replaces[long_short].items():
                    if correct_sent.find(llong) != -1:
                        wrong_sent = correct_sent.replace(llong, short)
                        return wrong_sent
            elif r == 2:
                for key, v in replaces[different].items():
                    if correct_sent.find(key) != -1:
                        wrong_sent = correct_sent.replace(key, v)
                        return wrong_sent
                    
        return wrong_sent

    def construct_spacing_mistakes(self, correct_sent):
        different = 'different'
        replaces = {
            different: {' de asemenea ': ' deasmenea ', ' de fapt ': ' defapt ', ' de altfel ': ' dealtfel ', ' nu mai ': ' numai ',
            ' numai ': ' nu mai ', ' o dată ': ' odată ', ' odată ': ' o dată ', ' în continuu ': ' încontinuu '}
        }

        for k, v in replaces.items():
            for kk, vv in replaces.items():
                nkk = self.replace_space_with_comma(kk)
                nvv = self.replace_space_with_comma(vv)
                replaces[k][nkk] = nvv
                
        wrong_sent = None
       
        for key, v in replaces[different].items():
            if correct_sent.find(key) != -1:
                wrong_sent = correct_sent.replace(key, v)
                return wrong_sent
        return wrong_sent

    def construct_coordinate_possesion_mistake(self, correct_sent):
        different = 'different'
        
        replaces = {
            different: {' de asemenea ': ' deasmenea ', ' de fapt ': ' defapt ', ' de altfel ': ' dealtfel ', ' nu mai ': ' numai ',
            ' numai ': ' nu mai ', ' o dată ': ' odată ', ' odată ': ' o dată ', ' în continuu ': ' încontinuu '}
        }

        for k, v in replaces.items():
            for kk, vv in replaces.items():
                nkk = self.replace_space_with_comma(kk)
                nvv = self.replace_space_with_comma(vv)
                replaces[k][nkk] = nvv
                
        wrong_sent = None
       
        for key, v in replaces[different].items():
            if correct_sent.find(key) != -1:
                wrong_sent = correct_sent.replace(key, v)
                return wrong_sent
        return wrong_sent

    def eliminate_one_comma(self, correct_sent):

        commas = Counter(list(correct_sent))[","]
        if commas == 0: return None
        r = random.randint(0, commas - 1)
        lcorrect_sent = list(correct_sent)
        all_ind = [i.start() for i in re.finditer(',', correct_sent)]
        del lcorrect_sent[all_ind[r]]
        return "".join(lcorrect_sent)

    def adjust_dash(self, correct_sent):
        """ eliminate dash """
        type_dash = random.randint(0, 1)
        if type_dash == 0:
            all_diacs = "".join([d for d, _ in CorpusGenerator.DIACS.items()])
            pattern = '[A-Za-z' + all_diacs + ']+-' + '[A-Za-z' + all_diacs + ']'

            matches = re.findall(pattern, correct_sent)
            if len(matches) == 0:
                return None
            else:
                match = matches[0]
            modified = match.replace('-', '')
            return correct_sent.replace(match, modified)
        else:
            """ add dash """
            # for no_dash_word, dash_word in self.dash_words.items():
            #     pattern = " " + no_dash_word + " "
            #     matches = re.findall(pattern, correct_sent)
            #     if len(matches) > 0:
            #         match = matches[0]
            # return correct_sent.replace(match, no_dash_word)

    def reconstruct_sent(self, wrong_words):
        reconstructed_sent = ' '.join(wrong_words)
        replaces = {" ,": ",", " !": "!", " ?": "?", " .": ".", ' - ': '-', ' :': ':',  ' ;': ';'}
        for k, v in replaces.items():
            reconstructed_sent = reconstructed_sent.replace(k, v)
        return reconstructed_sent

    def pronoun_form(self, tokens):
        pron_form_i = ['însumi', 'însuți', 'însuși', 'înșine', 'înșivă', 'înșiși', 'însămi', 'însăți',
                        'însăși', 'însene', 'însevă', 'înseși', 'însele']
        wrong_words = [token.text for token in tokens]
        for i, token in enumerate(tokens):
            if token.text.lower() in pron_form_i:
                index = random.randint(0, len(pron_form_i) - 1)
                wrong_words[i] = pron_form_i[index]
                return self.reconstruct_sent(wrong_words)
        return None

    def construct_mistake(self, tokens, correct_sent, wronged_sents, mistake_type):
        
        line = []
        wrong_sent = None

        """mistakes implying modfying more than one word"""
        if mistake_type is MistakeType.PRE:
            wrong_sent = self.construct_prepositions_mistakes(correct_sent)
        elif mistake_type is MistakeType.VIR:
            wrong_sent = self.eliminate_one_comma(correct_sent)
        elif mistake_type is MistakeType.CR:
            wrong_sent = self.adjust_dash(correct_sent)
        elif mistake_type is MistakeType.SP:
            wrong_sent = self.construct_spacing_mistakes(correct_sent)
        elif mistake_type is MistakeType.PI:
            wrong_sent = self.pronoun_form(tokens)
        else:
            tries = 0
            """ one word mistakes """
            while True:
                index = random.randint(0, len(tokens) - 1)
                #if self.is_wrongable_pos(tokens[index].pos) TODO keep this
                new_word = self.modify_word(tokens[index], mistake_type)
                if new_word is not None:
                    wrong_words = [token.text for token in tokens]
                    wrong_words[index] = new_word
                    wrong_sent = self.reconstruct_sent(wrong_words)
                    break
                tries += 1
                if tries > 20:  break
        
        if wrong_sent is not None:
            line.append(wrong_sent)
            line.append(correct_sent)
            line.append(mistake_type)
            wronged_sents.append(line)

    def wrong_sentence(self, correct_sent):
        "wrong the sentence, trying each case"
        correct_sent = self.clean_text(correct_sent)
        docs_ro = Document(Lang.RO, correct_sent)
        tokens = docs_ro.get_tokens()
        if self.filter_sentence(correct_sent, tokens) == False: return []
        nr_mistakes_types = len(MistakeType)
        wronged_sents = []

        for r in range(nr_mistakes_types):
            line = []
            self.construct_mistake(tokens, correct_sent, wronged_sents, mistake_type=MistakeType.TYPO)
            self.construct_mistake(tokens, correct_sent, wronged_sents, mistake_type=MistakeType.DIAC)
            self.construct_mistake(tokens, correct_sent, wronged_sents, mistake_type=MistakeType.PRE)
            self.construct_mistake(tokens, correct_sent, wronged_sents, mistake_type=MistakeType.VIR)
            self.construct_mistake(tokens, correct_sent, wronged_sents, mistake_type=MistakeType.CR)
            self.construct_mistake(tokens, correct_sent, wronged_sents, mistake_type=MistakeType.SP)
            self.construct_mistake(tokens, correct_sent, wronged_sents, mistake_type=MistakeType.PI)
        return wronged_sents
        
    def generate_based_on_features(self):
        global args
        lines = []

        for ffile in self.files:
            if len(lines) > args.samples:   break
            for i, correct_sent in enumerate(self.split_sentences(ffile)):
                if len(lines) % 100 == 0:   print('sent {} generated'.format(len(lines)))
                if len(lines) > args.samples:   break
                wrong_sents = self.wrong_sentence(correct_sent)
                for sent in wrong_sents:    lines.append(sent)
        for line in lines:
            print(line[0], '\t', line[1], line[2], '\t', file=log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--output', dest='output', action='store', default="out.csv")
    parser.add_argument('--task', dest='task', action='store', default="i")
    parser.add_argument('--rb_path', dest='rb_path', action='store', default="../readerbenchpy/")
    parser.add_argument('--samples', dest='samples', action='store', default=50, type=int)
    parser.add_argument('--type', dest='type', action='store', default=0, type=int)
    parser.add_argument('--feature_based', dest='feature_based', action='store_true', default=True)
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
    if args.feature_based:
        corpusGenerator.pre_computations()
        corpusGenerator.generate_based_on_features()
    #corpusGenerator.generate()
    #corpusGenerator.test_parser()
