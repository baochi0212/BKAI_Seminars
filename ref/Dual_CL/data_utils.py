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
def split_data(json_dict, max_sample=300):
    """
    take the sample from the json_dict with max_sample per class
    """
    data = []
    data_stat = {}
    for i in range(len(json_dict)):
        if json_dict[i]['label'] not in data_stat:
            data_stat[json_dict[i]['label']] = 0
        if data_stat[json_dict[i]['label']] < max_sample:
            data.append(json_dict[i])
            data_stat[json_dict[i]['label']] += 1 

    return data

def dataset2json(filename, proportion=0.1, max_sample=500):
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
    train_stat, test_stat = {}, {}
    if max_sample:
        proportion = 1
    else:
        max_sample = 1e+9
    for mode in ['train', 'validation', 'test']:
        length = int(len(dataset[mode])//(1/proportion)) if mode != "test" else len(dataset[mode])
        for i in range(length):
            sentence, label = dataset[mode][i]['sentence'], dataset[mode][i]['sentiment']
            if label not in labels:
                labels.append(label)
                locals()[f"train_stat"][label] = 0
                locals()[f"test_stat"][label] = 0
            if mode != 'test':
                if train_stat[label] < max_sample:
                    train_dict.append({'text': sentence, 'label': label})
                    train_stat[label] += 1
            else: 
                test_dict.append({'text': sentence, 'label': label})
                test_stat[label] += 1

    print("train:", len(train_dict), train_stat)
    print("validation:", len(test_dict), test_stat)


        
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
        train_data = split_data(train_data, max_sample=100)
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
    elif dataset == 'sensitive':
        train_data = json.load(open(os.path.join(data_dir, 'sensitive_train.json'), 'r', encoding='utf-8'))
        test_data = json.load(open(os.path.join(data_dir, 'sensitive_test.json'), 'r', encoding='utf-8'))
        label_dict = {'insult': 0, 'religion': 1, 'terrorism': 2, 'politics': 3, 'neutral': 4}
        # train_data = train_data[:int(len(train_data)//10)+1]
     

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

    dataset2json(filename, proportion=1, max_sample=400)
