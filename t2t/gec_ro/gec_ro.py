import re

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
import os

@registry.register_problem
class GecRo(text_problems.Text2TextProblem):

  @property
  def approx_vocab_size(self):
    return 2**14  # ~16k

  @property
  def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return False

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    # 10% evaluation data
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 9,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del tmp_dir
    del dataset_split

    files = os.listdir(data_dir)

    for ff in files:
      print(ff)
      with open(os.path.join(data_dir, ff), 'r', encoding='utf-8',  errors='replace') as f:
        for i, line in enumerate(f):
          #line = re.sub("[^a-z]+", " ", line.strip().lower())
          # print(line)
          if i % 3 == 0:
            source = line.strip()
          elif i % 3 == 1:
            target = line.strip()
          else:
            yield {
              "inputs": source,
              "targets": target,
            }