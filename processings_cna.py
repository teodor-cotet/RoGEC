import csv, os, argparse
from sklearn.model_selection import train_test_split
from typing import List 
from nltk.tokenize import word_tokenize

def split_file(in_file):
    samples = [] 
    with open(in_file, newline='') as csvfile:
        cvs_reader = csv.reader(csvfile)
        for row in cvs_reader:
            samples.append([row[0].strip(), row[1].strip()])
    train, test_dev = train_test_split(samples, shuffle=True, test_size=0.3)
    dev, test = test_dev = train_test_split(test_dev, shuffle=True, test_size=0.5)
    return train, dev, test

def write_to_csv(file_path, data):

    with open(file_path, 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in data:
            csv_writer.writerow(row)

def make_dirs(dirs: List[str]):
    for dir_name in dirs:
        if os.path.exists(dir_name) == False:
            os.makedirs(dir_name)

def write_txt_combined(file_path, samples):
    with open(file_path, 'w')  as f:
        for sample in samples:
            # first correct then wronged
            f.write(sample[1])
            f.write('\n')
            f.write(sample[0])
            f.write('\n')

def write_txt_single(file_path, name, samples):
     with open(os.path.join(file_path, name) + '_gold.txt', 'w') as gold,\
        open(os.path.join(file_path, name) + '_wronged.txt', 'w') as wronged:
        for sample in samples:
            # first correct then wronged
            gold.write(sample[1])
            gold.write('\n')
            wronged.write(sample[0])
            wronged.write('\n')

def do_split():
    global args 
    train_path = os.path.join(args.path_raw, 'train')
    dev_path = os.path.join(args.path_raw, 'dev')
    test_path = os.path.join(args.path_raw, 'test')

    make_dirs([train_path, dev_path, test_path])

    train_added, dev_added, test_added = split_file(os.path.join(args.path_raw, 'all', 'added.csv'))
    train_sent, dev_sent, test_sent = split_file(os.path.join(args.path_raw, 'all', 'cna_sent.csv'))
    train_phrase, dev_phrase, test_phrase = split_file(os.path.join(args.path_raw, 'all', 'cna_phrase.csv'))

    train_combined = train_added + train_phrase + train_sent
    dev_combined = dev_added + dev_phrase + dev_sent
    test_combined = test_added + test_phrase + test_sent

    write_to_csv(os.path.join(dev_path, 'dev_added.csv'), dev_added)
    write_to_csv(os.path.join(dev_path, 'dev_sent.csv'), dev_sent)
    write_to_csv(os.path.join(dev_path, 'dev_phrase.csv'), dev_phrase)
    write_to_csv(os.path.join(dev_path, 'dev_combined.csv'), dev_combined)

    write_to_csv(os.path.join(test_path, 'test_added.csv'), test_added)
    write_to_csv(os.path.join(test_path, 'test_sent.csv'), test_sent)
    write_to_csv(os.path.join(test_path, 'test_phrase.csv'), test_phrase)
    write_to_csv(os.path.join(test_path, 'test_combined.csv'), test_combined)

    # write txt, one per line gold/wronged
    write_txt_single(dev_path, 'dev_added', dev_added)
    write_txt_single(dev_path, 'dev_sent', dev_sent)
    write_txt_single(dev_path, 'dev_phrase', dev_phrase)
    write_txt_single(dev_path, 'dev_combined', dev_combined)

    # write txt, one per line gold/wronged
    write_txt_single(test_path, 'test_added', test_added)
    write_txt_single(test_path, 'test_sent', test_sent)
    write_txt_single(test_path, 'test_phrase', test_phrase)
    write_txt_single(test_path, 'test_combined', test_combined)
   
    # write txt together in the same file all combined
    write_txt_combined(os.path.join(train_path, 'train_combined.txt'), train_combined)
    write_txt_combined(os.path.join(dev_path, 'dev_combined.txt'), dev_combined)
    write_txt_combined(os.path.join(test_path, 'test_combined.txt'), test_combined)

def tokenize_file(in_path, out_path, name):

    tar = os.path.join(out_path, name)
    with open(os.path.join(in_path, name) + '.csv') as csvfile, open(tar + '_gold_tokenized.txt', 'w') as gold,\
         open(tar + '_wronged_tokenized.txt', 'w') as wronged:
        cvs_reader = csv.reader(csvfile)
        for row in cvs_reader:
            gold.write(" ".join(word_tokenize(row[1].strip())))
            gold.write('\n')
            wronged.write(" ".join(word_tokenize(row[0].strip())))
            wronged.write('\n')

def do_prepare_errant():

    all_path = os.path.join(args.path_raw, 'all')
    target_dir = os.path.join(all_path, 'tokenized')
    make_dirs([target_dir])

    tokenize_file(all_path, target_dir, 'added')
    tokenize_file(all_path,  target_dir, 'cna_phrase')
    tokenize_file(all_path, target_dir, 'cna_sent')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_raw', dest='path_raw', 
            action='store', default='corpora/cna/', type=str)
    parser.add_argument('-do_split', dest='do_split', 
            action='store_true')
    parser.add_argument('-prepare_errant', dest='prepare_errant', 
            action='store_true')
    args = parser.parse_args()

    if args.do_split:
        do_split()
    elif args.prepare_errant:
        do_prepare_errant()
