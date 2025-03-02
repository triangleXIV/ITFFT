import torch.utils.data as Data
from torch.utils.data import dataset
from tqdm import tqdm
import pickle
import numpy as np
import os
import logging

'''boxes_utils.py'''
from boxes_utils import *
from transformers import RobertaTokenizer, RobertaModel

from torchvision import transforms  # ++
from PIL import Image  # ++


class MyDataset(Data.Dataset):
    def __init__(self, data_dir, imagefeat_dir, tokenizer,
                 max_seq_len=64, crop_size=224, max_target_anp_len=100):
        self.imagefeat_dir = imagefeat_dir
        self.tokenizer = tokenizer  # Roberta
        self.relation_label_list = self.get_relation_labels()  # relevance label 0 or 1
        self.sentiment_label_list = self.get_sentiment_labels()  # sentiment labels 0, 1 or 2
        self.max_seq_len = max_seq_len  # maximum sequence length
        # self.max_GT_boxes = max_GT_boxes
        self.examples = self.creat_examples(data_dir)
        self.number = len(self.examples)
        # self.num_roi_boxes = num_roi_boxes
        # self.img_feat_dim = img_feat_dim

        self.crop_size = crop_size  # ++
        self.max_target_anp_len = max_target_anp_len  # ++

    def __len__(self):
        return self.number

    def __getitem__(self, index):
        line = self.examples[index]
        return self.transform(line, index)

    def creat_examples(self, data_dir):
        with open(data_dir, "rb") as f:
            dict = pickle.load(f)  # Open pkl
        examples = []
        for key, value in tqdm(dict.items(), desc="CreatExample"):  # desc passes in the str type as the progress bar title (similar to a description)
            examples.append(value)
        return examples

    def get_sentiment_labels(self):
        return ["0", "1", "2"]

    def get_relation_labels(self):
        return ["0", "1"]

    def transform(self, line, index):
        max_seq_len = self.max_seq_len
        # max_GT_boxes = self.max_GT_boxes
        # num_roi_boxes = self.num_roi_boxes

        crop_size = self.crop_size  # ++
        max_target_anp_len = self.max_target_anp_len  # ++

        # ++  # transforms.Compose is viewed as a container that can combine multiple data transforms at the same time
        transform = transforms.Compose([
            transforms.RandomCrop(crop_size),  # args.crop_size, by default it is set to be 224  # 224×224 pixels
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        '''
        {'iid': '16_05_05_780', 
        'sentence': 'Moorhead announces $T$ as new boys basketball coach # Spuds @ inforum', 
        'aspect': 'Bormann', 
        'sentiment': '1', 
        'relation': '1', 
        'boxes': [(24, 13, 242, 258)]}
        ---------------------------------------------
        {'iid': '1860693', 
        'sentence': 'RT @ ltsChuckBass : $T$ is everything # MCM', 
        'aspect': 'Chuck Bass', 
        'sentiment': '2', 
        'relation': None, 
        'adj': 'young successful sexy classic happy', 
        'noun': 'artist business men style kids'
        'caption': 'A man in a blue jacket and blue tie holding a cell phone.'}
        '''

        value = line  # per data set in pkl

        # text
        # Roberta
        text_a = value['sentence']
        text_b = value['aspect']

        # input_ids = self.tokenizer(text_a.lower(), text_b.lower())['input_ids']  # <s>text_a</s></s>text_b</s>

        # context_ids = self.tokenizer(text_a.lower())['input_ids']  # ++  # context
        token_a = self.tokenizer.tokenize(text_a.lower())
        tokens_a = ["[CLS]"] + token_a + ["[SEP]"]
        context_ids = self.tokenizer.convert_tokens_to_ids(tokens_a)
        context_mask = [1] * len(context_ids)  # ++

        # target_ids = self.tokenizer(text_b.lower())['input_ids']  # ++  # target
        token_b = self.tokenizer.tokenize(text_b.lower())
        tokens_b = ["[CLS]"] + token_b + ["[SEP]"]
        target_ids = self.tokenizer.convert_tokens_to_ids(tokens_b)
        target_mask = [1] * len(target_ids)  # ++

        tokens = tokens_a + token_b + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        added_input_mask = [1] * (len(input_ids) + 49)  # ++  # 1 or 49 is for encoding regional image representations

        padding_id = [1] * (max_seq_len - len(input_ids))  # <pad>: 1
        padding_mask = [0] * (max_seq_len - len(input_ids))

        context_padding_id = [1] * (max_seq_len - len(context_ids))  # ++
        context_padding_mask = [0] * (max_seq_len - len(context_ids))  # ++

        target_padding_id = [1] * (max_target_anp_len - len(target_ids))  # ++
        target_padding_mask = [0] * (max_target_anp_len - len(target_ids))  # ++

        input_ids += padding_id
        input_mask += padding_mask

        context_ids += context_padding_id  # ++
        context_mask += context_padding_mask  # ++

        target_ids += target_padding_id  # ++
        target_mask += target_padding_mask  # ++

        added_input_mask += padding_mask  # ++

        tokens = self.tokenizer.decode(input_ids)

        context_tokens = self.tokenizer.decode(context_ids)  # ++

        target_tokens = self.tokenizer.decode(target_ids)  # ++

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(context_ids) == max_seq_len
        assert len(context_mask) == max_seq_len


        # picture
        img_id = 0
        a_img_id = value['iid']
        # a_GT_boxes = value['boxes']
        img_id = (a_img_id + str(' ') * (16 - len(a_img_id))).encode('utf-8')

        image_path = os.path.join(self.imagefeat_dir, a_img_id + '.jpg')  # ++  # 17_06_10389.jpg

        count = 0  # ++

        if not os.path.exists(image_path):  # ++
            print(image_path)
        try:
            # def image_process
            image = image_process(image_path, transform)  # image reading
        except:
            count += 1
            # print('image has problem!')
            image_path_fail = os.path.join(self.imagefeat_dir, '17_06_4705.jpg')  # image not found
            image = image_process(image_path_fail, transform)

        '''
        GT_boxes = np.zeros((max_GT_boxes, 4))
        if a_GT_boxes:
            GT_boxes[:len(a_GT_boxes), :] = a_GT_boxes[:max_GT_boxes]  ######

        roi_boxes = np.zeros((num_roi_boxes, 4))
        img_feat = np.zeros((num_roi_boxes, self.img_feat_dim))
        spatial_feat = np.zeros((num_roi_boxes, 5))  ######

        # def read_npz
        a_img_feat, a_spatial_feat, a_roi_boxes, img_shape = read_npz(self.imagefeat_dir, a_img_id)
        assert len(a_roi_boxes) == len(a_img_feat)  # assert is used to judge an expression, triggering an exception if the expression condition is false.
        assert len(a_img_feat) == len(a_spatial_feat)
        current_num = len(a_img_feat)

        roi_boxes[:current_num, :] = a_roi_boxes
        img_feat[:current_num, :] = a_img_feat
        spatial_feat[:current_num, :] = a_spatial_feat

        'boxes_utils.py'
        box_label = get_attention_label(roi_boxes, GT_boxes)  # box_label: [max_GT_boxes, NUM_ROI_BOXES]  target: [max_GT_boxes, NUM_ROI_BOXES, 4]
        '''


        # relation
        relation_label = -1
        sentiment_label = -1
        # enumerate() combines traversable data objects (such as lists, tuples, or strings) into an indexed sequence that lists both data and data subscripts
        relation_label_map = {label: i for i, label in enumerate(self.relation_label_list)}
        sentiment_label_map = {label: i for i, label in enumerate(self.sentiment_label_list)}

        rel = value['relation']
        if rel:
            if rel == '2':
                rel = '1'
            relation_label = relation_label_map[rel]
        senti = value['sentiment']
        if senti:
            sentiment_label = sentiment_label_map[senti]


        # ++  # anp
        text_c = value['adj']
        text_d = value['noun']

        # ++  # caption
        text_e = value['caption']

        if text_c and text_d and text_e:
            text_cs = []
            for text_i in text_c.split():
                text_cs.append(text_i)

            tokens_cc = ["[CLS]"] + self.tokenizer.tokenize(text_cs[0].lower())
            tokens_cc += ["[SEP]"] + self.tokenizer.tokenize(text_cs[1].lower())
            tokens_cc += ["[SEP]"] + self.tokenizer.tokenize(text_cs[2].lower())
            tokens_cc += ["[SEP]"] + self.tokenizer.tokenize(text_cs[3].lower())
            tokens_cc += ["[SEP]"] + self.tokenizer.tokenize(text_cs[4].lower()) + ["[SEP]"]

            text_ds = []
            for text_j in text_d.split():
                text_ds.append(text_j)

            tokens_dc = ["[CLS]"] + self.tokenizer.tokenize(text_ds[0].lower())
            tokens_dc += ["[SEP]"] + self.tokenizer.tokenize(text_ds[1].lower())
            tokens_dc += ["[SEP]"] + self.tokenizer.tokenize(text_ds[2].lower())
            tokens_dc += ["[SEP]"] + self.tokenizer.tokenize(text_ds[3].lower())
            tokens_dc += ["[SEP]"] + self.tokenizer.tokenize(text_ds[4].lower()) + ["[SEP]"]

            # ++
            tokens_cdc = ["[CLS]"] + self.tokenizer.tokenize(text_cs[0].lower()) + self.tokenizer.tokenize(text_ds[0].lower())
            tokens_cdc += ["[SEP]"] + self.tokenizer.tokenize(text_cs[1].lower()) + self.tokenizer.tokenize(text_ds[1].lower())
            tokens_cdc += ["[SEP]"] + self.tokenizer.tokenize(text_cs[2].lower()) + self.tokenizer.tokenize(text_ds[2].lower())
            tokens_cdc += ["[SEP]"] + self.tokenizer.tokenize(text_cs[3].lower()) + self.tokenizer.tokenize(text_ds[3].lower())
            tokens_cdc += ["[SEP]"] + self.tokenizer.tokenize(text_cs[4].lower()) + self.tokenizer.tokenize(text_ds[4].lower()) + ["[SEP]"]

            # ++
            text_ec = text_e[:128]

            # adj_ids = self.tokenizer(text_cc)['input_ids']  # ++  # adj
            adj_ids = self.tokenizer.convert_tokens_to_ids(tokens_cc)
            adj_mask = [1] * len(adj_ids)

            # noun_ids = self.tokenizer(text_dc)['input_ids']  # ++  # noun
            noun_ids = self.tokenizer.convert_tokens_to_ids(tokens_dc)
            noun_mask = [1] * len(noun_ids)

            # anp_ids = self.tokenizer(text_cdc)['input_ids']  # ++  # anp
            anp_ids = self.tokenizer.convert_tokens_to_ids(tokens_cdc)
            anp_mask = [1] * len(anp_ids)

            token_ec = self.tokenizer.tokenize(text_ec.lower())
            tokens_ec = ["[CLS]"] + token_ec + ["[SEP]"]
            # caption_ids = self.tokenizer(text_ec.lower())['input_ids']  # ++  # caption
            caption_ids = self.tokenizer.convert_tokens_to_ids(tokens_ec)
            caption_mask = [1] * len(caption_ids)  # ++

            adj_padding_id = [1] * (max_target_anp_len - len(adj_ids))  # ++
            adj_padding_mask = [0] * (max_target_anp_len - len(adj_ids))  # ++

            noun_padding_id = [1] * (max_target_anp_len - len(noun_ids))  # ++
            noun_padding_mask = [0] * (max_target_anp_len - len(noun_ids))  # ++

            anp_padding_id = [1] * (max_target_anp_len - len(anp_ids))  # ++
            anp_padding_mask = [0] * (max_target_anp_len - len(anp_ids))  # ++

            caption_padding_id = [1] * (max_seq_len - len(caption_ids))  # ++
            caption_padding_mask = [0] * (max_seq_len - len(caption_ids))  # ++

            adj_ids += adj_padding_id  # ++
            adj_mask += adj_padding_mask  # ++

            noun_ids += noun_padding_id  # ++
            noun_mask += noun_padding_mask  # ++

            anp_ids += anp_padding_id  # ++
            anp_mask += anp_padding_mask  # ++

            caption_ids += caption_padding_id  # ++
            caption_mask += caption_padding_mask  # ++

            adj_tokens = self.tokenizer.decode(adj_ids)  # ++

            noun_tokens = self.tokenizer.decode(noun_ids)  # ++

            anp_tokens = self.tokenizer.decode(anp_ids)  # ++

            caption_tokens = self.tokenizer.decode(caption_ids)  # ++

            assert len(caption_ids) == max_seq_len
            assert len(caption_mask) == max_seq_len
        else:
            adj_tokens = None
            noun_tokens = None
            anp_tokens = None
            caption_tokens = None
            adj_ids = None
            noun_ids = None
            anp_ids = None
            caption_ids = None
            adj_mask = None
            noun_mask = None
            anp_mask = None
            caption_mask = None


        # tokens
        # input_ids: map input words to dictionary IDs in the model
        # input_mask: [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # sentiment_label: [0, 1, 2]
        # img_id: iid
        # img_shape: [img_h, img_w]
        # relation_label: [0, 1]
        # GT_boxes: boxes
        # roi_boxes: bbox
        # img_feat: x
        # spatial_feat
        # box_label

        # ++  # image: image read and adjusted to 224 x 224 pixels
        # ++  # context_tokens: context
        # ++  # context_ids
        # ++  # context_mask
        # ++  # target_tokens: target
        # ++  # target_ids
        # ++  # target_mask
        # ++  # added_input_mask
        # ++  # adj_tokens: adj
        # ++  # adj_ids
        # ++  # adj_mask
        # ++  # noun_tokens: noun
        # ++  # noun_ids
        # ++  # noun_mask
        # ++  # anp_tokens: anp
        # ++  # anp_ids
        # ++  # anp_mask
        # ++  # caption_tokens: caption
        # ++  # caption_ids
        # ++  # caption_mask

        if adj_ids != None and noun_ids != None and caption_ids != None:  # SA
            return tokens, context_tokens, target_tokens, input_ids, context_ids, target_ids, \
                   input_mask, context_mask, target_mask, added_input_mask, sentiment_label, \
                   img_id, image, relation_label, adj_tokens, noun_tokens, anp_tokens, caption_tokens, \
                   adj_ids, noun_ids, anp_ids, caption_ids, adj_mask, noun_mask, anp_mask, caption_mask
        else:  # VG
            return tokens, context_tokens, target_tokens, input_ids, context_ids, target_ids, \
                   input_mask, context_mask, target_mask, added_input_mask, sentiment_label, \
                   img_id, image, relation_label


# --
def read_npz(imagefeat_dir, img_id):
    if 'twitter' in imagefeat_dir.lower():
        feat_dict = np.load(os.path.join(imagefeat_dir, img_id + '.jpg.npz'))  ######

    # npy file is a numpy-specific binary file containing data in a matrix format that can be used to store an image

    img_feat = feat_dict['x']  # [2048, 100]
    img_feat = img_feat.transpose((1, 0))  # Rows and columns are swapped
    img_feat = (img_feat/np.sqrt((img_feat**2).sum()))

    # num_bbox = feat_dict['num_bbox']

    # boxes spatial
    bbox = feat_dict['bbox']
    img_h = feat_dict['image_h']
    img_w = feat_dict['image_w']
    # def get_spatial_feat
    spatial_feat = get_spatial_feat(bbox, img_h, img_w)

    return img_feat, spatial_feat, bbox, [float(img_h), float(img_w)]


# --
def get_spatial_feat(bbox, img_h, img_w):
    spatial_feat = np.zeros((bbox.shape[0], 5), dtype=np.float)  # shape[0] is the number of rows
    spatial_feat[:, 0] = bbox[:, 0]/float(img_w)
    spatial_feat[:, 1] = bbox[:, 1]/float(img_h)
    spatial_feat[:, 2] = bbox[:, 2]/float(img_w)
    spatial_feat[:, 3] = bbox[:, 3]/float(img_h)
    spatial_feat[:, 4] = (bbox[:, 2] - bbox[:, 0])*(bbox[:, 3] - bbox[:, 1]) / float(img_h*img_w)
    return spatial_feat


# ++
def image_process(image_path, transform):
    image = Image.open(image_path).convert('RGB')  # image reading
    image = transform(image)  # 224×224 pixels
    return image