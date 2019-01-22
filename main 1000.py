import sys
sys.path.append('../')
import datasets
import log_reg
from dataproc import extract_wvs
from dataproc import get_discharge_summaries
from dataproc import concat_and_split
from dataproc import build_vocab
from dataproc import vocab_index_descriptions
from dataproc import word_embeddings
from constants import MIMIC_3_DIR, DATA_DIR

import numpy as np
import pandas as pd

from collections import Counter, defaultdict
import csv
import math
import operator

print (MIMIC_3_DIR)
# Let's do some data processing in a much better way, with a notebook.
# 
# First, let's define some stuff.

# In[9]:

fname = '%s/notes_labeled.csv' % MIMIC_3_DIR
base_name = "%s/1000_disch" % MIMIC_3_DIR #for output
tr, dv, te = concat_and_split.split_data(fname, base_name=base_name)


# ## Build vocabulary from training data

# In[27]:

vocab_min = 0
vname = '%s/vocab_1000.csv' % MIMIC_3_DIR
build_vocab.build_vocab(vocab_min, tr, vname)


# ## Sort each data split by length for batching

# In[28]:

for splt in ['train', 'dev', 'test']:
    filename = '%s/1000_disch%s_split.csv' % (MIMIC_3_DIR, splt)
    df = pd.read_csv(filename)
    df['length'] = df.apply(lambda row: len(str(row['TEXT']).split()), axis=1)
    df = df.sort_values(['length'])
    df.to_csv('%s/1000_%s_full.csv' % (MIMIC_3_DIR, splt), index=False)

# In[32]:

Y = 50


# In[33]:

#first calculate the top k
counts = Counter()
dfnl = pd.read_csv('%s/notes_labeled_multigram.csv' % MIMIC_3_DIR)
for row in dfnl.itertuples():
    for label in str(row[4]).split(';'):
        counts[label] += 1


# In[34]:

codes_50 = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)


# In[35]:

codes_50 = [code[0] for code in codes_50[:Y]]


# In[37]:

with open('%s/TOP_%s_CODES.csv' % (MIMIC_3_DIR, str(Y)), 'w') as of:
    w = csv.writer(of)
    for code in codes_50:
        w.writerow([code])


# In[38]:

for splt in ['train', 'dev', 'test']:
    print(splt)
    hadm_ids = set()
    with open('%s/%s_full_hadm_ids.csv' % (MIMIC_3_DIR, splt), 'r') as f:
        for line in f:
            hadm_ids.add(line.rstrip())
    with open('%s/notes_labeled_multigram.csv' % MIMIC_3_DIR, 'r') as f:
        with open('%s/multi_%s_%s.csv' % (MIMIC_3_DIR, splt, str(Y)), 'w') as of:
            r = csv.reader(f)
            w = csv.writer(of)
            #header
            w.writerow(next(r))
            i = 0
            for row in r:
                hadm_id = row[1]
                if hadm_id not in hadm_ids:
                    continue
                codes = set(str(row[3]).split(';'))
                filtered_codes = codes.intersection(set(codes_50))
                if len(filtered_codes) > 0:
                    w.writerow(row[:3] + [';'.join(filtered_codes)])
                    i += 1


# In[39]:

for splt in ['train', 'dev', 'test']:
    filename = '%s/multi_%s_%s.csv' % (MIMIC_3_DIR, splt, str(Y))
    df = pd.read_csv(filename)
    df['length'] = df.apply(lambda row: len(str(row['TEXT']).split()), axis=1)
    df = df.sort_values(['length'])
    df.to_csv('%s/multi_%s_%s.csv' % (MIMIC_3_DIR, splt, str(Y)), index=False)

