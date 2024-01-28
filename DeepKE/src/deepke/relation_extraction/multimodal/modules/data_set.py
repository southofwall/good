from torch.utils.data import Dataset
import logging
import os
import ast
from PIL import Image
import json

logger = logging.getLogger(__name__)
WORKING_PATH= "../../../../../example/re/multimodal/data"

class MyDataset(Dataset):
    def __init__(self, mode, text_name, limit=None):
        self.text_name = 'txt'
        #  'twitter_stream_2018_07_23_13_0_2_218.jpg': {'text': 'RT @astarrynight _ She was the face of Marc Jacobs Beauty at age 64 . God works hard but Jessica Lange works harder .',
        #   'label': 'None'},
        self.data = self.load_data(mode, limit)
        self.image_ids=list(self.data.keys())
        #  'twitter_stream_2018_07_23_13_0_2_218.jpg'
        for id in self.data.keys():
            self.data[id]["image_path"] = os.path.join(WORKING_PATH,"img_org",mode,str(id))
    
    def load_data(self, mode, limit=None):
        # lujin
        data_set=dict()
        load_file=os.path.join(WORKING_PATH, self.text_name ,"ours_"+mode+".txt")
        logger.info("Loading data from {}".format(load_file))
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # image,sentence,label=[],[],[]
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)   # str to dict
                # image.append(line['img_id'])
                # # token变成一个句子
                # sentence.append(sentence(line['token']))
                # label.append(line['relation'])

            # for data in datas:
            #     image = data['image_id']
            #     sentence = data['text']
            #     label = data['label']
                if os.path.isfile(os.path.join(WORKING_PATH,"img_org",mode,str(line['img_id']))):
                    data_set[str(line['img_id'])]={"text":self.sentence(line['token']), 'label': line['relation']}
        return data_set


    def image_loader(self,id):
        return Image.open(self.data[id]["image_path"])
    def text_loader(self,id):
        return self.data[id]["text"]


    def __getitem__(self, index):
        id=self.image_ids[index]
        text = self.text_loader(id)
        image_feature = self.image_loader(id)
        label = self.data[id]["label"]
        return text,image_feature, label, id

    def __len__(self):
        return len(self.image_ids)
    @staticmethod
    def collate_func(batch_data):
        batch_size = len(batch_data)
 
        if batch_size == 0:
            return {}

        text_list = []
        image_list = []
        label_list = []
        id_list = []
        for instance in batch_data:
            text_list.append(instance[0])
            image_list.append(instance[1])
            label_list.append(instance[2])
            id_list.append(instance[3])
        return text_list, image_list, label_list, id_list

    # token2sentence
    def sentence(self,token_list):
        # 创建一个新的空字符串用于构建句子
        sentence = ''

        sentence = ' '.join(token_list)
        sentence = sentence.replace(' :', '')
        return sentence  # 输出整合后的句子

