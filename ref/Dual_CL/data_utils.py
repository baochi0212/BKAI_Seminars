import os
import json
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import fuckit
from sklearn.model_selection import train_test_split
#if import dataset
with fuckit:
    from datasets import load_dataset
filename = "uit-nlp/vietnamese_students_feedback"
# dataset = load_dataset(filename)

data_dir = "./data"
def split_data(json_dict):
    """
    split the json_dict
    """
    pass
def dataset2json(filename, proportion=0.1):
    '''
    for dataset library 
    experiment with varied proportion of data
    '''

    dataset = load_dataset(filename)
    filename = filename.split('/')[0]
    train_path = f"./data/{filename}_Train.json"
    test_path = f"./data/{filename}_Test.json"
    label_path = f"./data/{filename}_label.txt"
    train_dict, test_dict, labels = [], [], []
    for mode in ['train', 'validation', 'test']:
        length = int(len(dataset[mode])//(1/proportion) + 1) if mode == 'train' else len(dataset[mode])
        for i in range(length):
            sentence, label = dataset[mode][i]['sentence'], dataset[mode][i]['sentiment']
            if label not in labels:
                labels.append(label)
            if mode == 'train':
                train_dict.append({'text': sentence, 'label': label})
            else: 
                test_dict.append({'text': sentence, 'label': label})
            
        
    json.dump(train_dict, open(train_path, 'w'), indent=3, ensure_ascii=False)
    json.dump(test_dict, open(test_path, 'w'), indent=3, ensure_ascii=False)
    with open(label_path, 'w') as f:
        for label in labels:
            f.write(str(label) + '\n')
        

    

#static(args), cls(cls, args) -> both called in class i.o object, while static more strict
def text2dict(filename):
    if 'phoATIS' in filename:
        label_dict = {'[UNK]': 0}
        idx = 1
    else:
        label_dict = {}
        idx = 0 
    with open(f"{data_dir}/{filename}", 'r') as f:
        for line in f.readlines():
            line = line.strip()
            #convert to special tokens

            if line != "UNK":
                line = '[' + line + ']'
                label_dict[line] = idx 
                idx += 1 

    return label_dict



class MyDataset(Dataset):

    def __init__(self, raw_data, label_dict, tokenizer, model_name, method):
        label_list = list(label_dict.keys()) if method not in ['ce', 'scl'] else []
        sep_token = ['[SEP]'] if model_name in ['bert', 'phobert'] else ['</s>']
        dataset = list()
        for data in raw_data:
            tokens = data['text'].lower().split(' ')
            label = '[' + str(data['label']) + ']' if '[' in list(label_dict.keys())[0] else str(data['label'])
            if  label not in label_dict:
                label_id = 0
            else:
                label_id = label_dict[label]

            dataset.append((label_list + sep_token + tokens, label_id))
        self._dataset = dataset
    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


def my_collate(batch, tokenizer, method, num_classes):
    tokens, label_ids = map(list, zip(*batch))
    text_ids = tokenizer(tokens,
                         padding=True,
                         truncation=True,
                         max_length=256,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    if method not in ['ce', 'scl']:
        positions = torch.zeros_like(text_ids['input_ids'])
        positions[:, num_classes:] = torch.arange(0, text_ids['input_ids'].size(1)-num_classes)
        text_ids['position_ids'] = positions
    return text_ids, torch.tensor(label_ids)


def load_data(dataset, data_dir, tokenizer, train_batch_size, test_batch_size, model_name, method, workers):
    if dataset == 'sst2':
        train_data = json.load(open(os.path.join(data_dir, 'SST2_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SST2_Test.json'), 'r', encoding='utf-8'))
        train_data = train_data[:int(len(train_data)//10)+1]
        
        label_dict = {'positive': 0, 'negative': 1}

    elif dataset == 'trec':
        train_data = json.load(open(os.path.join(data_dir, 'TREC_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'TREC_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'description': 0, 'entity': 1, 'abbreviation': 2, 'human': 3, 'location': 4, 'numeric': 5}
    elif dataset == 'cr':
        train_data = json.load(open(os.path.join(data_dir, 'CR_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'CR_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'subj':
        train_data = json.load(open(os.path.join(data_dir, 'SUBJ_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'SUBJ_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'subjective': 0, 'objective': 1}
    elif dataset == 'pc':
        train_data = json.load(open(os.path.join(data_dir, 'procon_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'procon_Test.json'), 'r', encoding='utf-8'))
        label_dict = {'positive': 0, 'negative': 1}
    elif dataset == 'phoATIS':

        train_data = json.load(open(os.path.join(data_dir, 'phoATIS_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'phoATIS_Test.json'), 'r', encoding='utf-8'))
        label_dict = text2dict('phoATIS_label.txt')
    elif dataset == 'uit-nlp':
        train_data = json.load(open(os.path.join(data_dir, 'uit-nlp_Train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'uit-nlp_Test.json'), 'r', encoding='utf-8'))
        label_dict = text2dict('uit-nlp_label.txt')
     

    else:
        raise ValueError('unknown dataset')
    trainset = MyDataset(train_data, label_dict, tokenizer, model_name, method)
    testset = MyDataset(test_data, label_dict, tokenizer, model_name, method)
    collate_fn = partial(my_collate, tokenizer=tokenizer, method=method, num_classes=len(label_dict))
    train_dataloader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn, pin_memory=True)
    test_dataloader = DataLoader(testset, test_batch_size, shuffle=False, num_workers=workers, collate_fn=collate_fn, pin_memory=True)
    return train_dataloader, test_dataloader

if __name__ == '__main__':
    # tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    # train_data = json.load(open(os.path.join(data_dir, 'uit-nlp_Train.json'), 'r', encoding='utf-8'))
    # test_data = json.load(open(os.path.join(data_dir, 'uit-nlp_Test.json'), 'r', encoding='utf-8'))
    # label_dict = dict([('[' + str(key) + ']', key) for key in [0, 1, 2]])
    # special_tokens = {"additional_special_tokens": list(label_dict.keys())}
    # tokenizer.add_special_tokens(special_tokens)
    # print(len(tokenizer.get_vocab()))
    # # # for word in tokenizer.get_vocab():
    # # #     if "unused" in word:
    # # #         print(word)
    # # #         break

    # dataset2json(filename)
