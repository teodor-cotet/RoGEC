import os
import sys
import string
import argparse
from collections import Counter
from statistics import mean, stdev
from nltk.tokenize import sent_tokenize

DIACS, COMMON_ABBR = None, None

def construct_globals():
    
    global DIACS, COMMON_ABBR
    DIACS = "ĂÂÎȘȚăâîșț"
    COMMON_ABBR = set(['lb.', 'ex.', 'nr.', 'Hr.', 'hr.', 'sec.', 'cca.', 'ed.', 'vol.', 'pag.',
         ' p.', ' d.', 'a.k.a.', 'cf.', 'n.r.', 'id.', 'coord.', 'lat.', 'Ed.', 'Dvs.', 'dvs.',
         'C.F.R.', 'Al.', 'etc.', 'dj.', ' n.', 'St.', 'Sf.', 'trad.', '(.', 'ar.', 'e.c.',
         'gr.', 'aprox.', 'art.', 'sysop.', 'art.', 'ș.a.', 'î.e.n.', 'Vol.', 'www.', 's.d.', ' a.'
         'pg.', 'pp.', 'str.', 'Bd.', 'Sos.', 'jud.', 'Dr.', 'ha.'])
    UPPER = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    UPPER += list("ĂÂÎȘȚ")
    for elem in UPPER: COMMON_ABBR.add(elem + '.')

log = open('corpora/sentences_wiki.txt', 'wt')

def ratio_diacritics(counter: Counter):
    global DIACS
    diac_set = set(DIACS)
    sum_diacs = sum([counter[d] for d in diac_set])
    sum_rest = sum(counter[d] for d in counter if d not in diac_set)
    value = sum_diacs / (sum_rest + 1e-7)
    return value

def ratio_normal_characters(counter: Counter):
    global DIACS
    ascii_printable = string.printable
    normal_chars = set(ascii_printable + DIACS)
    sum_normal = sum([counter[c] for c in normal_chars])
    sum_rest = sum(counter[c] for c in counter if c not in normal_chars)
    ratio_normal = sum_rest / (sum_normal + 1e-7)
    return ratio_normal

def compute_statistics_text(content: str):
    counter = Counter(content)
    return ratio_diacritics(counter), ratio_normal_characters(counter)

def get_txt_files(dir_path: str):
    entries = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
    txt_files = [f for f in entries if os.path.isfile(f) and f.endswith('.txt')]
    return txt_files

def compute_statistics():
    global args
    txt_files = get_txt_files(args.dir_path)
    diac_values, normal_values = [], []
    
    for ffile in txt_files:
        with open(ffile, 'rt', encoding='utf-8', errors='replace') as f:
            content = f.read()
            diac_v, normal_v = compute_statistics_text(content)
            diac_values.append(diac_v)
            normal_values.append(normal_v)

    diac_m, diac_std = mean(diac_values), stdev(diac_values)
    normal_m, normal_std = mean(normal_values), stdev(normal_values)
    
    print(f'diac threshold: {diac_m - 0.6 * diac_std}')
    print(f'normal threshold: {normal_m + normal_std}')

def filter_sentence(sent: str):
    global COMMON_ABBR
    try:
        end_chars = set(list(".?!"))
        counter = Counter(sent)
        if (counter['|'] == 0 and len(sent) > 8 and counter['\n'] == 0 and
            (sent[0].isupper() or sent[0] == '"') 
            and counter['"'] % 2 == 0 and sent[-1] in end_chars and sent[-2] != ' ' 
            and sent.find('www.') == -1 and not sent.startswith('-ului') and
            ((counter['('] + counter[')']) % 2 == 0 and (counter['[]'] + counter[']']) % 2 == 0)):
            good_ending = True
            for abbr in COMMON_ABBR:    good_ending = good_ending and (not sent.endswith(abbr))
            return good_ending
        return False
    except:
        return False

def generate_sentences():
    global args

    txt_files = get_txt_files(args.dir_path)
    gen_sents = set()
    for ffile in txt_files:
            with open(ffile, 'rt', encoding='utf-8', errors='replace') as f:
                for content in f:
                    try:
                        diac_ratio, ratio_normal = compute_statistics_text(content)
                        if diac_ratio < 0.01:
                            continue
                        if ratio_normal > 0.025:
                            continue
                        sentences = sent_tokenize(content)
                        sentences = [sent.strip() for sent in sentences]
                        filtered_sents = [sent for sent in sentences if filter_sentence(sent)]

                        for s in filtered_sents:
                            if s not in gen_sents:
                                gen_sents.add(s)
                                print(s, file=log)
                        
                        if len(gen_sents) > 5e5:
                            gen_sents = set()
                    except:
                        print('error')


if __name__ == "__main__":
    # Define and parse program input
    parser = argparse.ArgumentParser(description="Filter sentences from wikipedia")
    parser.add_argument("-dir_path", default='/opt/teo/gec/corpora/', help="Path to txt files", type=str)
    parser.add_argument("-stats", action='store_true', help="To compute statistics")

    construct_globals()
    if args.stats:
        compute_statistics()
    elif args.wiki:
        generate_sentences()