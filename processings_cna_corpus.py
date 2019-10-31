import sys
import csv, os, argparse
import logging
sys.path.insert(0, '../readerbenchpy')

from rb.similarity.vector_model import VectorModelType, CorporaEnum, VectorModel
from rb.similarity.vector_model_instance import VECTOR_MODELS
from rb.core.lang import Lang
from rb.core.document import Document
from rb.core.pos import POS
from enum import Enum
from utils import clean_diacs

logging.getLogger().setLevel(logging.INFO)

class MistakeEnum(Enum):
    SIN = 'sintaxă'
    ORT = 'ortografie'
    PUNCT = 'punctuație'
    MORPH = 'morfologie'
    SEM = 'semantică'
    LEX = 'lexic'
    PRON1 = 'ortoepie'
    PRON2 = 'pronunțare'
    GRAPH = 'grafie'
    STIL = 'stilistică'
    TEHN = 'tehnoredactare'
    PLEON = 'pleonasm'
    FON = 'fonetică'
    OTHER = 'other'

class RestrictedMistakeEnum(Enum):
    SINTAX = 'sintaxă'
    ORTOGRAPHY = 'ortografie'
    PUNCTUATION = 'punctuație'
    SEMANTIC = 'semantică'
    LEXIC = 'lexic'
    STIL = 'stilistică'
    OTHER = 'nespecificat'

mistakes_types_mappings = {
    MistakeEnum.SIN: RestrictedMistakeEnum.SINTAX,
    MistakeEnum.ORT: RestrictedMistakeEnum.ORTOGRAPHY,
    MistakeEnum.PUNCT: RestrictedMistakeEnum.PUNCTUATION,
    MistakeEnum.MORPH: RestrictedMistakeEnum.ORTOGRAPHY,
    MistakeEnum.SEM: RestrictedMistakeEnum.SEMANTIC,
    MistakeEnum.LEX: RestrictedMistakeEnum.LEXIC,
    MistakeEnum.GRAPH: RestrictedMistakeEnum.ORTOGRAPHY,
    MistakeEnum.STIL: RestrictedMistakeEnum.STIL,
    MistakeEnum.TEHN: RestrictedMistakeEnum.ORTOGRAPHY,
    MistakeEnum.PLEON: RestrictedMistakeEnum.SEMANTIC,
    MistakeEnum.OTHER: RestrictedMistakeEnum.OTHER
}

class CNACorpus():

    def __init__(self):
        pass

    """ split corpus into phrases and sentences """
    def split_corpus(self):
        global args
        vector_model = VECTOR_MODELS[Lang.RO][CorporaEnum.README][VectorModelType.WORD2VEC](
                    name=CorporaEnum.README.value, lang=Lang.RO)
        cna_csv_all = csv.reader(open(args.path_to_cna_csv, 'rt', encoding='utf-8'))
        samples_sentences, samples_phrases = [], []

        for i, row in enumerate(cna_csv_all):
            if i == 0:  continue
            logging.info(f'Sample {i}')
            if len(row[1]) < 3 or len(row[2]) < 3: continue
            row = [clean_diacs(r) for r in row]
            doc1 = Document(lang=Lang.RO, text=row[1], vector_model=vector_model)
            is_sentence = False
            for word in doc1.get_words():
                if word.pos is POS.VERB:
                    is_sentence = True
                    break
            doc2 = Document(lang=Lang.RO, text=row[3], vector_model=vector_model)
            mistake_type = None
            
            found = False
            for word in doc2.get_words():
                for mistake in MistakeEnum:
                    if mistake.value == word.lemma.lower():
                        mistake_type = mistake
                        found = True
                        break
                if found:   break
            if mistake_type is None:    mistake_type = MistakeEnum.OTHER
            if mistake_type in mistakes_types_mappings:
                mistake_type = mistakes_types_mappings[mistake_type]
            else:
                continue

            if is_sentence:
                samples_sentences.append([row[1], row[2], mistake_type.value, row[3]])
            else:
                samples_phrases.append([row[1], row[2], mistake_type.value, row[3]])

        # with open(args.path_to_cna_phrase_csv, 'wt', encoding='utf-8') as csv_phrase:
        #         csv_writer = csv.writer(csv_phrase)
        #         csv_writer.writerows(samples_phrases)            

        with open(args.path_to_cna_sent_csv, 'wt', encoding='utf-8') as csv_sent:
                csv_writer = csv.writer(csv_sent)
                csv_writer.writerows(samples_sentences) 

    
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
            doc1 = Document(lang=Lang.RO, text=row[0])
            doc2 = Document(lang=Lang.RO, text=row[1])
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

    """ filter samples which are 
        1. identical wrong/corrected
        2. indentical with others
    """
    def filter_corpus(self, path_to_old_file: str, path_to_new_file: str):
        global args
        cna_csv = csv.reader(open(path_to_old_file, 'rt', encoding='utf-8'))
        samples_pairs, samples_wrong = set(), set()
        filtered = []
        nr_duplicates_pairs, nr_dupplicates_wrong, nr_same = 0, 0, 0
        stats_per_type = { mistake_type.value: 0 for mistake_type in RestrictedMistakeEnum }
        del stats_per_type['nespecificat']
        stats_per_type['other'] = 0

        for i, row in enumerate(cna_csv):
            row[0] = row[0].strip()
            row[1] = row[1].strip()
            sample_pair = (row[0], row[1])

            keep = True
            if sample_pair in samples_pairs:
                nr_duplicates_pairs += 1
                keep = False
            samples_pairs.add(sample_pair)
            
            if row[0] == row[1]:
                nr_same += 1
                keep = False

            if row[0] in samples_wrong and keep:
                nr_dupplicates_wrong += 1
            samples_wrong.add(row[0])

            if keep:
                mistake_type = row[2].strip()
                if mistake_type in stats_per_type:
                    stats_per_type[mistake_type] += 1
                else:
                    logging.info(f"{i} with {mistake_type}")
                if mistake_type == "other":
                    row[2] = RestrictedMistakeEnum.OTHER.value
                filtered.append(row)
                
        logging.info(f"duplicates wrong: {nr_dupplicates_wrong}")
        logging.info(f"duplicates pairs (eliminated): {nr_duplicates_pairs}")
        logging.info(f"same wrong/correct: {nr_same}")
        logging.info(f"stats per type: {stats_per_type}")

        with open(path_to_new_file, 'wt', encoding='utf-8') as csv_sent:
            csv_writer = csv.writer(csv_sent)
            csv_writer.writerows(filtered) 
        
    def prepare_for_errant_format(self, path_csv_file: str, 
            path_to_origin_file: str, path_to_correct_file: str):  
        global args
        logging.info(f'file: {path_csv_file}')
        cna_csv = csv.reader(open(path_csv_file, 'rt', encoding='utf-8'))
        with open(path_to_origin_file, 'wt') as origin, open(path_to_correct_file, 'wt') as corrected:
            for i, row in enumerate(cna_csv):
                doc = Document(lang=Lang.RO, text=row[0])
                tokens = [token.text for token in doc.get_words()]
                origin.write(" ".join(tokens) + "\n")

                doc = Document(lang=Lang.RO, text=row[1])
                tokens = [token.text for token in doc.get_words()]
                corrected.write(" ".join(tokens) + "\n")
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_cna_csv', dest='path_to_cna_csv', action='store', default='corpora/cna.csv', type=str)
    parser.add_argument('--path_to_cna_sent_csv', dest='path_to_cna_sent_csv', 
            action='store', default='corpora/cna_sent_2.csv', type=str)
    parser.add_argument('--path_to_cna_phrase_csv', dest='path_to_cna_phrase_csv', 
            action='store', default='corpora/cna_phrase.csv', type=str)
    parser.add_argument('--path_to_new_cna_phrase_csv', dest='path_to_new_cna_phrase_csv', 
            action='store', default='corpora/cna_filt_phrase.csv', type=str)
    parser.add_argument('--path_to_new_cna_sent_csv', dest='path_to_new_cna_sent_csv', 
            action='store', default='corpora/cna_filt_sent.csv', type=str)
    parser.add_argument('--path_errant_sent_original', dest='path_errant_sent_original', 
            action='store', default='corpora/errant_sent_original.txt', type=str)
    parser.add_argument('--path_errant_sent_corrected', dest='path_errant_sent_corrected', 
            action='store', default='corpora/errant_sent_corrected.txt', type=str)
    parser.add_argument('--path_errant_phrase_original', dest='path_errant_phrase_original', 
            action='store', default='corpora/errant_phrase_original.txt', type=str)
    parser.add_argument('--path_errant_phrase_corrected', dest='path_errant_phrase_corrected', 
            action='store', default='corpora/errant_phrase_corrected.txt', type=str)
    
    parser.add_argument('--filter', dest='filter', action='store_true', default=False)
    parser.add_argument('--stats', dest='stats', action='store_true', default=False)
    parser.add_argument('--errant', dest='errant', action='store_true', default=False)
    
    args = parser.parse_args()

    for k in args.__dict__:
        if args.__dict__[k] is not None:
            print(k, '->', args.__dict__[k])

    cna_corpus = CNACorpus()
    if args.filter:
        cna_corpus.filter_corpus(path_to_old_file=args.path_to_cna_sent_csv,
                                path_to_new_file=args.path_to_new_cna_sent_csv)
        cna_corpus.filter_corpus(path_to_old_file=args.path_to_cna_phrase_csv,
                                path_to_new_file=args.path_to_new_cna_phrase_csv)
    elif args.stats:
        cna_corpus.compute_stats_corpus(path_to_file=args.path_to_new_cna_sent_csv)
        cna_corpus.compute_stats_corpus(path_to_file=args.path_to_new_cna_phrase_csv)
    elif args.errant:
        cna_corpus.prepare_for_errant_format(path_csv_file=args.path_to_new_cna_phrase_csv,
                                             path_to_origin_file=args.path_errant_phrase_original,
                                             path_to_correct_file=args.path_errant_phrase_corrected)