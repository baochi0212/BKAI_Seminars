import os
import sys
import json

data_dir = '/home/xps/educate/code/hust/Lab/Contrastive_learning_survey-/data'



def json_phoATIS(path=f'{data_dir}/processed/IDSF/phoATIS'):
    '''
    change ATIS format to json for text cls
    '''
    train_path = f"{path}/phoATIS_Train.json"
    test_path = f"{path}/phoATIS_Test.json"
    # if not os.path.exists(train_path):
    #     os.system(f"touch {train_path}")
    # if not os.path.exists(test_path):
    #     os.system(f"touch {test_path}")
    train_data = []
    test_data = []
    for mode in ['train', 'dev', 'test']:

        
        lines = [line.strip() for line in open(f"{path}/{mode}/seq.in").readlines()]
        labels = [label.strip() for label in open(f"{path}/{mode}/label").readlines()]
        for line, label in zip(lines, labels):
            if mode != 'train':
                test_data.append(dict([('text', line), ('label', label)]))
            else:
                train_data.append(dict([('text', line), ('label', label)]))
    print(train_data[0])
    print(test_data[0])
    json.dump(train_data, open(train_path, 'w', encoding='utf8'), indent=3, ensure_ascii=False)
    json.dump(test_data, open(test_path, 'w', encoding='utf8'), indent=3, ensure_ascii=False)


if __name__ == '__main__':
    json_phoATIS()
                
            
        



