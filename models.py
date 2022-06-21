import torch
from torch import nn
from transformers import BertForSequenceClassification, BertConfig

class BERT(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim, number_of_classes):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        configuration = BertConfig(hidden_size=4, num_attention_heads=4, num_hidden_layers=4, intermediate_size=4, num_labels=number_of_classes)
        self.bert = BertForSequenceClassification(configuration)
        
    def forward(self, s, n):
        s = self.embeddings(s)
        n = torch.unsqueeze(n, 2)

        x = torch.cat((s, n), dim=2)

        return self.bert(inputs_embeds=x)
    
class ANN(nn.Module):
    def __init__(self, n_inputs, number_of_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(n_inputs),
            nn.Linear(n_inputs, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, number_of_classes)
        )
        def weights_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.classifier.apply(weights_init)

    def forward(self, X):
        probs = self.classifier(X)
        return probs
    
class COMB(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_dim, number_of_classes):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        configuration = BertConfig(hidden_size=4, num_attention_heads=4, num_hidden_layers=4, intermediate_size=4, num_labels=4)
        self.bert = BertForSequenceClassification(configuration)
        self.num_classifier = nn.Sequential(
            nn.BatchNorm1d(num_dim),
            nn.Linear(num_dim, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4)
        )
        self.final = nn.Sequential(
            nn.BatchNorm1d(8),
            nn.Linear(8, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, number_of_classes)
        )
        
    def forward(self, cigar, counts, numerical):
        s = self.embeddings(cigar)
        n = torch.unsqueeze(counts, 2)

        text = torch.cat((s, n), dim=2)
        text = self.bert(inputs_embeds=text)["logits"]
        
        numerical = self.num_classifier(numerical)
        
        final = torch.cat((text, numerical), dim=1)
        
        return self.final(final)
