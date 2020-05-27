# raw model
# only beam
echo "experiments on test for $1 $2 model"

#python3 transformer_bert.py --checkpoint=checkpoints/$1 --beam=8  --in_file_decode=corpora/cna/test/test_phrase_wronged.txt --out_file_decode=corpora/cna/test/$2_beam8_b/test_phrase_predicted.txt --decode_mode=True

python3 transformer_bert.py --checkpoint=checkpoints/$1 --beam=8  --in_file_decode=corpora/cna/test/test_phrase_wronged.txt --out_file_decode=corpora/cna/test/$2_beam8_b/test_phrase_predicted.txt --decode_mode=True --lm_path=corpora/30m_wiki_clean.arpa --use_bucket=True
python3 transformer_bert.py --checkpoint=checkpoints/$1 --beam=8  --in_file_decode=corpora/cna/test/test_added_wronged.txt --out_file_decode=corpora/cna/test/$2_beam8_b/test_added_predicted.txt --decode_mode=True --lm_path=corpora/30m_wiki_clean.arpa --use_bucket=True
python3 transformer_bert.py --checkpoint=checkpoints/$1 --beam=8  --in_file_decode=corpora/cna/test/test_sent_wronged.txt --out_file_decode=corpora/cna/test/$2_beam8_b/test_sent_predicted.txt --decode_mode=True --lm_path=corpora/30m_wiki_clean.arpa --use_bucket=True
python3 transformer_bert.py --checkpoint=checkpoints/$1 --beam=8  --in_file_decode=corpora/cna/test/test_combined_wronged.txt --out_file_decode=corpora/cna/test/$2_beam8_b/test_combined_predicted.txt --decode_mode=True --lm_path=corpora/30m_wiki_clean.arpa --use_bucket=True
# beam and lm unnormalized
python3 transformer_bert.py --checkpoint=checkpoints/$1 --beam=8  --in_file_decode=corpora/cna/test/test_phrase_wronged.txt --out_file_decode=corpora/cna/test/$2_beam8_lm_b/test_phrase_predicted.txt --decode_mode=True --lm_path=corpora/30m_wiki_clean.arpa --use_bucket=True --lm=True
python3 transformer_bert.py --checkpoint=checkpoints/$1 --beam=8  --in_file_decode=corpora/cna/test/test_added_wronged.txt --out_file_decode=corpora/cna/test/$2_beam8_lm_b/test_added_predicted.txt --decode_mode=True --lm_path=corpora/30m_wiki_clean.arpa --use_bucket=True --lm=True
python3 transformer_bert.py --checkpoint=checkpoints/$1 --beam=8  --in_file_decode=corpora/cna/test/test_sent_wronged.txt --out_file_decode=corpora/cna/test/$2_beam8_lm_b/test_sent_predicted.txt --decode_mode=True --lm_path=corpora/30m_wiki_clean.arpa --use_bucket=True --lm=True
python3 transformer_bert.py --checkpoint=checkpoints/$1 --beam=8  --in_file_decode=corpora/cna/test/test_combined_wronged.txt --out_file_decode=corpora/cna/test/$2_beam8_lm_b/test_combined_predicted.txt --decode_mode=True --lm_path=corpora/30m_wiki_clean.arpa --use_bucket=True --lm=True
# beam and lm normalized
python3 transformer_bert.py --checkpoint=checkpoints/$1 --beam=8  --in_file_decode=corpora/cna/test/test_phrase_wronged.txt --out_file_decode=corpora/cna/test/$2_beam8_lmn_b/test_phrase_predicted.txt --decode_mode=True --lm_path=corpora/30m_wiki_clean.arpa --use_bucket=True --lm=True normalize_lm=True
python3 transformer_bert.py --checkpoint=checkpoints/$1 --beam=8  --in_file_decode=corpora/cna/test/test_added_wronged.txt --out_file_decode=corpora/cna/test/$2_beam8_lmn_b/test_added_predicted.txt --decode_mode=True --lm_path=corpora/30m_wiki_clean.arpa --use_bucket=True --lm=True normalize_lm=True
python3 transformer_bert.py --checkpoint=checkpoints/$1 --beam=8  --in_file_decode=corpora/cna/test/test_sent_wronged.txt --out_file_decode=corpora/cna/test/$2_beam8_lmn_b/test_sent_predicted.txt --decode_mode=True --lm_path=corpora/30m_wiki_clean.arpa --use_bucket=True --lm=True normalize_lm=True
python3 transformer_bert.py --checkpoint=checkpoints/$1 --beam=8  --in_file_decode=corpora/cna/test/test_combined_wronged.txt --out_file_decode=corpora/cna/test/$2_beam8_lmn_b/test_combined_predicted.txt --decode_mode=True --lm_path=corpora/30m_wiki_clean.arpa --use_bucket=True --lm=True normalize_lm=True
# beam normalized and lm normalized
python3 transformer_bert.py --checkpoint=checkpoints/$1 --beam=8  --in_file_decode=corpora/cna/test/test_phrase_wronged.txt --out_file_decode=corpora/cna/test/$2_beam8_lmn_bn/test_phrase_predicted.txt --decode_mode=True --lm_path=corpora/30m_wiki_clean.arpa --use_bucket=True --lm=True normalize_lm=True --normalize_beam=True
python3 transformer_bert.py --checkpoint=checkpoints/$1 --beam=8  --in_file_decode=corpora/cna/test/test_added_wronged.txt --out_file_decode=corpora/cna/test/$2_beam8_lmn_bn/test_added_predicted.txt --decode_mode=True --lm_path=corpora/30m_wiki_clean.arpa --use_bucket=True --lm=True normalize_lm=True --normalize_beam=True
python3 transformer_bert.py --checkpoint=checkpoints/$1 --beam=8  --in_file_decode=corpora/cna/test/test_sent_wronged.txt --out_file_decode=corpora/cna/test/$2_beam8_lmn_bn/test_sent_predicted.txt --decode_mode=True --lm_path=corpora/30m_wiki_clean.arpa --use_bucket=True --lm=True normalize_lm=True --normalize_beam=True
python3 transformer_bert.py --checkpoint=checkpoints/$1 --beam=8  --in_file_decode=corpora/cna/test/test_combined_wronged.txt --out_file_decode=corpora/cna/test/$2_beam8_lmn_bn/test_combined_predicted.txt --decode_mode=True --lm_path=corpora/30m_wiki_clean.arpa --use_bucket=True --lm=True normalize_lm=True --normalize_beam=True
