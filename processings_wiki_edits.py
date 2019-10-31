import sys
import csv, os, argparse
import logging
sys.path.insert(0, '../readerbenchpy')

from rb.core.lang import Lang
from rb.core.document import Document
from rb.core.pos import POS
from enum import Enum
from utils import clean_diacs

logging.getLogger().setLevel(logging.INFO)

class WikiEdits():

    def __init__(self):
        pass

    def prepare_for_errant_format(self, path_wiki_file: str, 
            path_to_origin_file: str, path_to_correct_file: str):  
        global args
        logging.info(f'parsing file: {path_wiki_file}')

        with open(path_to_origin_file, 'wt') as origin, open(path_to_correct_file, 'wt') as corrected,\
                open(path_wiki_file, 'rt') as wiki:
            
            original_sents, corrected_sents = [], []
            for i, line in enumerate(wiki):
                if (i + 1) % 3 == 0:   continue
                if (i + 1) % 3 == 1: original_sents.append(line)
                if (i + 1) % 3 == 2: corrected_sents.append(line)

            for org, cor in zip(original_sents, corrected_sents):
                doc = Document(lang=Lang.RO, text=org)
                tokens = [token.text for token in doc.get_words()]
                origin.write(" ".join(tokens) + "\n")

                doc = Document(lang=Lang.RO, text=cor)
                tokens = [token.text for token in doc.get_words()]
                corrected.write(" ".join(tokens) + "\n")

    def compute_stats_corpus(self, path_to_file: str):
        global args
        logging.info(f'file: {path_to_file}')
        cna_csv = csv.reader(open(path_to_file, 'rt', encoding='utf-8'))
        tokens_wrong, tokens_correct = [], [] 
        stats_per_type = { mistake_type.value: 0 for mistake_type in RestrictedMistakeEnum }
        samples_nr = len(cna_csv)
        for i, row in enumerate(cna_csv):
            row[0] = row[0].strip()
            row[1] = row[1].strip()
            doc1 = Document(lang=Lang.RO, text=row[0], vector_model=vector_model)
            doc2 = Document(lang=Lang.RO, text=row[1], vector_model=vector_model)
            nr_ww, nr_wc = len(doc1.get_words()), len(doc2.get_words())
            for t1 in doc1.get_words():
                tokens_correct.append(t1.text)
            for t2 in doc2.get_words():
                tokens_wrong.append(t2.text)
            stats_per_type[row[2]] += 1

        unq_tokens_wrong, unq_tokens_right = set(tokens_correct), set(tokens_wrong)
        total_tokens_correct, total_tokens_wrong = sum(tokens_correct), sum(tokens_wrong)
        avg_tokens_wrong = len(tokens_correct) / samples_nr
        avg_tokens_correct = len(tokens_wrong) / samples_nr
        logging.info(f"stats per type: {stats_per_type}")
        logging.info(f"average nr tokens wrong samples {avg_tokens_wrong}")
        logging.info(f"average nr tokens correct samples {avg_tokens_correct}")
        logging.info(f"nr tokens wrong samples {len(tokens_wrong)}")
        logging.info(f"nr tokens correct samples {len(tokens_correct)}")
        logging.info(f"nr unq tokens correct samples {len(unq_tokens_right)}")
        logging.info(f"nr unq tokens wrong samples {len(unq_tokens_wrong)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_wiki_edits', dest='path_wiki_edits', 
            action='store', default='corpora/wiki_edits/ro_edits_100k_sample.txt', type=str)
    parser.add_argument('--path_errant_sent_original', dest='path_errant_sent_original', 
            action='store', default='corpora/wiki_edits/errant_sent_original.txt', type=str)
    parser.add_argument('--path_errant_sent_corrected', dest='path_errant_sent_corrected', 
            action='store', default='corpora/wiki_edits/errant_sent_corrected.txt', type=str)
    
    parser.add_argument('--prepare', dest='prepare', action='store_true', default=False)
    parser.add_argument('--errant', dest='errant', action='store_true', default=False)
    parser.add_argument('--stats', dest='stats', action='store_true', default=False)
    
    args = parser.parse_args()

    for k in args.__dict__:
        if args.__dict__[k] is not None:
            print(k, '->', args.__dict__[k])

    wiki_edits = WikiEdits()
    if args.prepare:
        logging.info('Prepare wiki edit corpus for errant')
        wiki_edits.prepare_for_errant_format(path_wiki_file=args.path_wiki_edits,
                                             path_to_origin_file=args.path_errant_sent_original,
                                             path_to_correct_file=args.path_errant_sent_corrected)