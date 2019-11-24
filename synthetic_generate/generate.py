import argparse
import aspell
import numpy as np
import logging
import random
import re
from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok
from typing import List
from rb.core.document import Document
from rb.core.lang import Lang
from rb.core.pos import POS


logging.getLogger().setLevel(logging.INFO)

MATCH_ALPHA_WORD, LOWER = None, None
speller, word_set, detokenizer = None, None, None

def construct_globals():
    global MATCH_ALPHA_WORD, LOWER, speller, word_set, detokenizer
    MATCH_ALPHA_WORD = "[A-Za-zĂÂÎȘȚăâîșț]+"
    LOWER = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    LOWER += list("ăâșîț")
    speller = aspell.Speller('lang', 'ro')
    word_set = set()
    detokenizer = Detok()

def quote_repl(matchobj) -> str:
    quote_char_start = matchobj.group(1).strip()[0]
    quote_char_end = matchobj.group(3).strip()[0]
    return quote_char_start + matchobj.group(2) + quote_char_end 

def point_repl(matchobj) -> str:
    return matchobj.group(1) + '. '

""" detokenizer for romanian """
def reconstruct_sentence(sent: List[str]) -> str:
    global detokenizer
    text = detokenizer.detokenize(sent)
    text = re.sub(r'(")\s+(.*?)\s+(")', quote_repl, text)
    text = re.sub(r'(«)\s+(.*?)\s+(»)', quote_repl, text)
    text = re.sub(r'(“)\s+(.*?)\s+(”)', quote_repl, text)
    text = re.sub(r'(„)\s+(.*?)\s+(”)', quote_repl, text)
    text = re.sub(r'\s*,\s*', ', ', text)
    text = re.sub(r'(\D)\s*\.\s*$', point_repl, text)
    text = re.sub(r'\s*\?\s*$', '? ', text)
    text = re.sub(r'\s*\-\s*', '-', text)
    text = re.sub(r'\s*\!\s*$', '! ', text)

    return text

def modify_words(tokenst: List[str]) -> List[str]:
    global args, speller, word_set, MATCH_ALPHA_WORD

    p_err = np.random.normal(loc=args.perr_m, scale=args.perr_stdev)
    p_err = max(p_err, 0)
    
    for t in tokenst:
        if t not in word_set and len(word_set) < 2e5:
            word_set.add(t.lower())
    idxs_valid_words = [i for i, t in enumerate(tokenst) if re.fullmatch(MATCH_ALPHA_WORD, t) is not None]
    words_changed = int(round(p_err * len(idxs_valid_words), 0))

    for _ in range(words_changed):
        # recompute ids
        idxs_valid_words = [i for i, t in enumerate(tokenst) if re.fullmatch(MATCH_ALPHA_WORD, t) is not None]
        idx = np.random.randint(0, len(idxs_valid_words), size=1)[0]
        original_idx = idxs_valid_words[idx]
        p_op = np.random.uniform()
        
        if p_op < args.p_subst:
            # subst
            confusion_set = speller.suggest(tokenst[original_idx])[:21]
            if tokenst[original_idx] in confusion_set:
                confusion_set.remove(tokenst[original_idx])
            else:
                confusion_set = confusion_set[:20]
            if len(confusion_set) > 0:
                id_subst = np.random.randint(0, len(confusion_set), size=1)[0]
                subst = confusion_set[id_subst]

                tokenst[original_idx] = subst
        elif p_op < args.p_subst + args.p_del:
            # del
            del tokenst[original_idx]
        elif p_op < args.p_subst + args.p_del + args.p_ins:
            # ins
            tokenst.insert(original_idx, random.sample(word_set, 1)[0])
        else:
            # swap adj
            if original_idx < len(tokenst) - 1:
                tokenst[original_idx], tokenst[original_idx + 1] = tokenst[original_idx + 1], tokenst[original_idx]
    return tokenst

def modify_chars(tokenst: List[str]) -> List[str]:
    global args, LOWER, MATCH_ALPHA_WORD
    idxs_valid_words = [i for i, t in enumerate(tokenst) if re.fullmatch(MATCH_ALPHA_WORD, t) is not None]
    chars_changed = int(round(1/10 * len(idxs_valid_words)))

    for _ in range(chars_changed):
        idxs_valid_words = [i for i, t in enumerate(tokenst) if re.fullmatch(MATCH_ALPHA_WORD, t) is not None]
        idx = np.random.randint(0, len(idxs_valid_words), size=1)[0]
        original_idx = idxs_valid_words[idx]
        p_op = np.random.uniform()
        if p_op < args.p_subst:
            # subst
            id_char = np.random.randint(0, len(tokenst[original_idx]), size=1)[0]
            tokenl = list(tokenst[original_idx])
            tokenl[id_char] = random.choice(LOWER)
            tokenst[original_idx] = "".join(tokenl)
        elif p_op < args.p_subst + args.p_del:
            # del
            id_char = np.random.randint(0, len(tokenst[original_idx]), size=1)[0]
            tokenl = list(tokenst[original_idx])
            del tokenl[id_char]
            tokenst[original_idx] = "".join(tokenl)
        elif p_op < args.p_subst + args.p_del + args.p_ins:
            # ins
            id_char = np.random.randint(0, len(tokenst[original_idx]), size=1)[0]
            tokenl = list(tokenst[original_idx])
            tokenl.insert(id_char, random.choice(LOWER))
            tokenst[original_idx] = "".join(tokenl)
        else:
            # swap adj
            id_char = np.random.randint(0, len(tokenst[original_idx]), size=1)[0]
            tokenl = list(tokenst[original_idx])
            if id_char < len(tokenl) - 1:
                tokenl[id_char], tokenl[id_char + 1] = tokenl[id_char + 1], tokenl[id_char]
            tokenst[original_idx] = "".join(tokenl)
    return tokenst

def modify_sentence(sent: str):
    global args, speller, word_set, LOWER, MATCH_ALPHA_WORD
   
    doc = Document(text=sent, lang=Lang.RO)
    tokens = [token for token in doc.get_words()]
    tokenst = [token.text for token in doc.get_words()]

    tokents = modify_words(tokenst)
    tokents = modify_chars(tokenst)
    
    return reconstruct_sentence(tokenst)

def generate_sentences():
    global args
    with open(args.sent_file_in, 'rt', encoding='utf-8', errors='replace') as fin,\
        open(args.sent_file_out, 'wt', encoding='utf-8') as fout:
        for sent in fin:
            sent = sent.strip()
            try:
                sent_modified = modify_sentence(sent)
            except:
                continue
            fout.write(sent + '\n')
            fout.write(sent_modified + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter sentences from wikipedia")
    parser.add_argument("-sent_file_in", default='/home/teo/projects/gec/corpora/clean_sentences/30mil_wiki_clean.txt',
                     help="Path to txt file", type=str)
    parser.add_argument("-sent_file_out", default='/home/teo/projects/gec/corpora/clean_sentences/30mil_wiki_dirty.txt',
                     help="Path to out txt file", type=str)
    parser.add_argument("-perr_m", default=0.15, help="Mean of normal distribution for p err", type=float)
    parser.add_argument("-perr_stdev", default=0.2, help="St dev of normal distribution for p err", type=float)
    parser.add_argument("-p_subst", default=0.7, help="Subsituing probablity", type=float)
    parser.add_argument("-p_del", default=0.1, help="Deleting probability", type=float)
    parser.add_argument("-p_ins", default=0.1, help="Inserting probability", type=float)
    parser.add_argument("-p_swap", default=0.1, help="Swapping probability", type=float)
    args = parser.parse_args()

    for k in args.__dict__:
        if args.__dict__[k] is not None:
            logging.info('{} -> {}'.format(k,  str(args.__dict__[k])))
    if abs(args.p_subst + args.p_del + args.p_ins + args.p_swap - 1) > 1e-5:
        logging.warning('Probabilities do not sum up to 1.')
    else:
        construct_globals()
        generate_sentences()
