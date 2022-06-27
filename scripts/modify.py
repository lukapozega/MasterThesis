from statistics import NormalDist
import csv
from Bio import SeqIO
from Bio.SeqIO import FastaIO
from Bio.Seq import Seq
from random import randint
import sys
import numpy as np
from multiprocessing import Pool

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class Distribution:
    def __init__(self, row):
        self.bucket_idx = int(row[0])
        self.M_ratio_mean = float(row[1])
        self.M_ratio_std = float(row[2])
        self.I_ratio_mean = float(row[3])
        self.I_ratio_std = float(row[4])
        self.D_ratio_mean = float(row[5])
        self.D_ratio_std = float(row[6])
        self.X_ratio_mean = float(row[7])
        self.X_ratio_std = float(row[8])
        self.bucket_start = int(row[9])
        self.bucket_end = int(row[10])
    
    def contains(self, middle):
        return middle >= self.bucket_start and middle <= self.bucket_end

with open(sys.argv[2], 'r') as f:
    reader = csv.reader(f)
    next(reader)
    data = list(reader)

distributions = list(map(lambda x: Distribution(x), data))

new_reads = []

for i, record in enumerate(SeqIO.parse(sys.argv[1], "fasta")):
    l = record.description.split(",")
    start = int(l[2].split("position=")[1].split()[0].split("-")[0])
    end = int(l[2].split("position=")[1].split()[0].split("-")[1])
    middle = (start + end) // 2
    modified = False
    for d in distributions:
        if (d.contains(middle)):
            # deletion
            # generator = NormalDist(mu = d.D_ratio_mean, sigma= d.D_ratio_std)
            var = d.D_ratio_std**2
            theta = var / d.D_ratio_mean
            k = d.D_ratio_mean / theta
            num_deletions = int(np.random.gamma(k, theta, 1)[0] * len(record) // 100)
            for i in range(num_deletions):
                pos = randint(0, len(record))
                record.seq = record.seq[:pos] + record.seq[pos+1:]

            # insertion
            # generator = NormalDist(mu = d.I_ratio_mean, sigma= d.I_ratio_std)
            var = d.I_ratio_std**2
            theta = var / d.I_ratio_mean
            k = d.I_ratio_mean / theta
            num_insertions = int(np.random.gamma(k, theta, 1)[0] * len(record) // 100)
            # num_insertions = max(int(generator.samples(1)[0] * len(record) // 100), 0)
            bases = ["A", "C", "G", "T"]
            for i in range(num_insertions):
                pos = randint(0, len(record))
                base = bases[randint(0, 3)]
                record.seq = record.seq[:pos] + base + record.seq[pos:]

            # mismatch
            var = d.X_ratio_std**2
            theta = var / d.X_ratio_mean
            k = d.X_ratio_mean / theta
            num_mismatch = int(np.random.gamma(k, theta, 1)[0] * len(record) // 100)
            bases = ["A", "C", "G", "T"]
            positions = {}
            for i in range(num_mismatch):
                pos = randint(0, len(record)-1)
                base = record.seq[pos]
                while (base == record.seq[pos]):
                    base = bases[randint(0, 3)]
                positions[pos] = base
            seq = list(str(record.seq))
            for k in positions.keys():
                seq[k] = positions[k]
            record.seq = "".join(seq)
            record.seq = Seq(record.seq)
            modified = True
            break
    if (not modified):
        eprint("Read " + record.id + " outside of buckets. Skipping...")
    else:
        new_reads.append(record)
        # print(">" + record.description, flush=True)
        # print(record.seq, flush=True)
    # break

with open(sys.argv[3], 'w') as f:
   fasta_out = FastaIO.FastaWriter(f, wrap=None) 
   fasta_out.write_file(new_reads)
# SeqIO.write(new_reads, sys.argv[3], "fasta")
