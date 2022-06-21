import torch

vocab2index_extended = {
    "<PAD>": 0,
    "X": 1,
    "I": 2,
    "D": 3,
    "=": 4,
}

vocab2index = {
    "<PAD>": 0,
    "M": 1,
    "I": 2,
    "D": 3,
}

def encode_cigar(cigar, extended_cigar):
    count = ""
    encoding = []
    for j in cigar:
        if j.isdigit():
            count += j
        else:
            index = vocab2index_extended[j] if extended_cigar else vocab2index[j]
            encoding.append((index, float(count)))
            count = ""
    return encoding

# with open("datasets/dataset19.txt", "r") as f:
#     lines = f.readlines()
#     count = 0
#     for line in lines:
#         count += len(encode_cigar(line.strip().split()[0]))
#     print(count/len(lines))

def pad_tensor(tensor, max_len, pad_index=0):
    seq = tensor
    tensor_size = len(seq)
    if tensor_size >= max_len:
        return seq[:max_len]
    return seq + [pad_index]*(max_len-tensor_size)


def pad_collate_fn(batch, pad_index=0):
    cigars, counts, overlap_len, similarity, labels = zip(*batch)
    # print(len(cigars), len(cigars[0]))
    lengths = torch.tensor([len(cigar) for cigar in cigars])
    # Process the text instances
    max_len = min(512,lengths.max().item())
    # max_len = lengths.max().item()
    cigars = torch.tensor([pad_tensor(cigar, max_len, pad_index) for cigar in cigars])
    counts = torch.tensor([pad_tensor(count, max_len, pad_index) for count in counts])
    # enc = torch.tensor([pad_tensor(cigar[0], max_len, pad_index) for cigar in cigars])
    # cou = torch.tensor([pad_tensor(cigar[1], max_len, pad_index) for cigar in cigars])
    labels = torch.tensor(labels)
    overlap_len = torch.tensor(overlap_len)
    similarity = torch.tensor(similarity)
    num = torch.transpose(torch.stack((overlap_len, similarity)), 0 , 1)
    return (cigars, counts, num), labels, lengths