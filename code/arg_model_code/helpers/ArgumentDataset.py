import torch

class ArgumentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None, features=None, sentences=None):
        self.encodings = encodings
        self.labels = labels
        self.features = features
        self.sentences = sentences  

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.features is not None:
            item['features'] = torch.tensor(self.features[idx], dtype=torch.float)

        if self.sentences is not None:
            item['sentence'] = self.sentences[idx]

        return item

    def __len__(self):
        if self.labels is not None:
            return len(self.labels)
        
        return len(self.encodings['input_ids'])
