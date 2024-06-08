# TODO tokenize captions first and then feed to the class Dataset.
# TODO document the processings and have comments in codes.
import torch

from text_processing import get_vocab, preprocess_txt

# Parameters
# Eva:
# PARAMS = {'batch_size': 64,
#           'shuffle': True,
#           'num_workers': 16}

# ozkan lab computer:
TRAIN_PARAMS = {'batch_size': 32,
          'shuffle': True,
          'num_workers': 6}

TEST_PARAMS = {'batch_size': 128,
          'shuffle': True,
          'num_workers': 6}

class TestDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, paths, ids, feature_folder):
        'Initialization'
        self.paths = paths
        self.ids = ids
        self.feature_folder = feature_folder

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.paths)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        name = self.paths[index] + ".pt"
        feature = torch.load(self.feature_folder / name, map_location='cpu')
        id = self.ids[index]

        return feature, id

class TrainDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, paths, captions, tokenizer, feature_folder, max_length=30):
        'Initialization'
        self.paths = paths
        self.captions = captions
        self.tokenizer = tokenizer
        self.feature_folder = feature_folder
        self.max_length = max_length

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.paths)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        name = self.paths[index] + ".pt"
        feature = torch.load(self.feature_folder / name, map_location='cpu')
        
        caption = self.captions[index]
        start_token = self.tokenizer.bos_token
        end_token = self.tokenizer.eos_token
        encoding = self.tokenizer.encode_plus(
            f"{start_token} {caption} {end_token}",
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        # The 'input_ids' are the tokenized representation of the caption
        input_ids = encoding['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask = encoding['attention_mask'].squeeze(0)  # Remove batch dimension
        #attention_mask = torch.ones(feature.shape[0], dtype=torch.long)
        return feature, input_ids, attention_mask

def get_loader_and_vocab(dt, tokenizer):
    train_data, val_data, test_data = dt.load_data()
    train_paths, train_captions, _ = zip(*train_data)
    train_dataset = TrainDataset(train_paths, train_captions, tokenizer, feature_folder=dt.train_features_folder)
    train_loader = torch.utils.data.DataLoader(train_dataset, **TRAIN_PARAMS)
    val_paths, val_ids = zip(*val_data)
    val_dataset = TestDataset(val_paths, val_ids, dt.val_features_folder)
    val_loader = torch.utils.data.DataLoader(val_dataset, **TEST_PARAMS)

    if test_data == None:
        test_loader = None
    else:
        test_paths, test_ids = zip(*test_data)
        test_dataset = TestDataset(test_paths, test_ids, dt.test_features_folder)
        test_loader = torch.utils.data.DataLoader(test_dataset, **TEST_PARAMS)
    return train_loader, val_loader, test_loader
