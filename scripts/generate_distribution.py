import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--paf", type=str, help="location of paf file with cigar strings")
parser.add_argument("--buckets_num", type=int, help="size of bucket for mean and std(default: 10000)", metavar="")
parser.add_argument("--output", type=str, help="location of output csv file", metavar="")
args = parser.parse_args()

if (not args.paf):
    raise ValueError("PAF file path not valid")

if (not args.output):
    raise ValueError("Output file path not valid")

if (not args.buckets_num):
    raise ValueError("Invalid number of buckets")

data = []
with open(args.paf, "r") as file:
    for i, line in enumerate(file):
        line = line.split()
        q_start = int(line[2])
        q_end = int(line[3])
        t_start = int(line[7])
        t_end = int(line[8])
        cigar = line[-1].split(":")[-1]
        cigar_dictionary = {"D": 0, "I": 0, "X": 0, "=": 0}
        b = ""
        for j in cigar:
            if j.isdigit():
                b += j
            else:
                cigar_dictionary[j] += int(b)
                b = ""
        data.append([q_start, q_end, t_start, t_end, cigar_dictionary["D"], cigar_dictionary["I"], cigar_dictionary["="], cigar_dictionary["X"]])

df = pd.DataFrame(data, columns=["q_start", "q_end", "t_start", "t_end", "D", "I", "M", "X"])

df["CIGAR_len"] = (df["M"] + df["I"] + df["D"])
df["M_ratio"] = (df["M"] / df["CIGAR_len"] * 100)
df["X_ratio"] = (df["X"] / df["CIGAR_len"] * 100)
df["I_ratio"] = (df["I"] / df["CIGAR_len"] * 100)
df["D_ratio"] = (df["D"] / df["CIGAR_len"] * 100)

df["t_middle"] = ((df["t_start"] + df["t_end"])//2)

max_end = df["t_end"].max()
bucket_size =  max_end // args.buckets_num

buckets = list(range(0, max_end + bucket_size, bucket_size))
df['bucket_idx'] = pd.cut(df['t_middle'], bins=buckets, labels=False)
df['bucket_idx'] = df['bucket_idx'].astype('int')

distributions = df[["M_ratio", "I_ratio", "D_ratio", "X_ratio", "bucket_idx"]].groupby("bucket_idx").agg([np.mean, np.std])
distributions.columns = [f'{i}_{j}' for i, j in distributions.columns]
distributions.reset_index(inplace=True)

buck = {v: k for v, k in enumerate(buckets)}
diff = buck[1] - buck[0]

distributions['bucket_start'] = distributions["bucket_idx"].map(buck)
distributions['bucket_end'] = distributions["bucket_idx"].map(lambda x: buck[x] + diff)

with open(args.output, "w") as file:
    file.write(distributions.to_csv(index=False))
