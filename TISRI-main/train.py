import os
import logging
import argparse  # Parsing command line arguments
import random
import datetime
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from itertools import cycle
from transformers import RobertaTokenizer, RobertaModel
from transformers import (
    BertModel,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)  # ++
from sklearn.metrics import precision_recall_fscore_support
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler

'''DataProcessor.py'''
from DataProcessor import *
'''model.py'''
from model import Coarse2Fine
'''boxes_utils.py'''
from boxes_utils import*
from optimization import BertAdam

import resnet.resnet as resnet  # ++
from resnet.resnet_utils import myResnet  # ++


# calculating F1 scores
def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro \
        = precision_recall_fscore_support(true, preds, average='macro')
    # f_macro = 2 * p_macro * r_macro / (p_macro + r_macro)
    return p_macro, r_macro, f_macro


# find predictions that are the same as true sentiment labels
def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)  # output the largest sentiment label within each array
    return np.sum(outputs == labels)  # summation of identical labels


'''
Warmup is a way to optimise for the learning rate, a method of warming up the learning rate mentioned in the ResNet paper.
at the beginning of training, a smaller learning rate is chosen, and after training some epochs, the learning rate is modified to a pre-set learning rate for training.
'''
def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


'''
# ++
def post_dataloader_VG(batch):
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')  # Defaults to cuda:0

    tokens, context_tokens, target_tokens, input_ids, context_ids, target_ids, \
    input_mask, context_mask, target_mask, added_input_mask, sentiment_label, \
    img_id, image, relation_label = batch

    input_ids = list(map(list, zip(*input_ids)))
    input_mask = list(map(list, zip(*input_mask)))
    # img_shape = list(map(list, zip(*img_shape)))

    context_ids = list(map(list, zip(*context_ids)))  # ++
    context_mask = list(map(list, zip(*context_mask)))  # ++

    target_ids = list(map(list, zip(*target_ids)))  # ++
    target_mask = list(map(list, zip(*target_mask)))  # ++

    added_input_mask = list(map(list, zip(*added_input_mask)))  # ++

    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)
    # img_shape = torch.tensor(img_shape, dtype=torch.float).to(device)

    context_ids = torch.tensor(context_ids, dtype=torch.long).to(device)  # ++
    context_mask = torch.tensor(context_mask, dtype=torch.long).to(device)  # ++

    target_ids = torch.tensor(target_ids, dtype=torch.long).to(device)  # ++
    target_mask = torch.tensor(target_mask, dtype=torch.long).to(device)  # ++

    added_input_mask = torch.tensor(added_input_mask, dtype=torch.long).to(device)  # ++

    image = image.to(device)  # ++ #

    sentiment_label = sentiment_label.to(device).long()
    relation_label = relation_label.to(device).long()

    return tokens, context_tokens, target_tokens, input_ids, context_ids, target_ids, \
           input_mask, context_mask, target_mask, added_input_mask, sentiment_label, \
           img_id, image, relation_label
'''


# store data into the device
def post_dataloader(batch):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # default is cuda:0
    # tokens, input_ids, input_mask, sentiment_label, \
    # img_id, img_shape, relation_label, GT_boxes, roi_boxes, img_feat, spatial_feat, box_labels = batch

    tokens, context_tokens, target_tokens, input_ids, context_ids, target_ids, \
    input_mask, context_mask, target_mask, added_input_mask, sentiment_label, \
    img_id, image, relation_label, adj_tokens, noun_tokens, anp_tokens, caption_tokens, \
    adj_ids, noun_ids, anp_ids, caption_ids, adj_mask, noun_mask, anp_mask, caption_mask = batch

    # list() is used to convert a tuple into a list
    # zip() returns a list of tuples
    input_ids = list(map(list, zip(*input_ids)))
    input_mask = list(map(list, zip(*input_mask)))
    # img_shape = list(map(list, zip(*img_shape)))

    context_ids = list(map(list, zip(*context_ids)))  # ++
    context_mask = list(map(list, zip(*context_mask)))  # ++

    target_ids = list(map(list, zip(*target_ids)))  # ++
    target_mask = list(map(list, zip(*target_mask)))  # ++

    added_input_mask = list(map(list, zip(*added_input_mask)))  # ++

    # create a tensor of the specified shape
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    input_mask = torch.tensor(input_mask, dtype=torch.long).to(device)
    # img_shape = torch.tensor(img_shape, dtype=torch.float).to(device)

    context_ids = torch.tensor(context_ids, dtype=torch.long).to(device)  # ++
    context_mask = torch.tensor(context_mask, dtype=torch.long).to(device)  # ++

    target_ids = torch.tensor(target_ids, dtype=torch.long).to(device)  # ++
    target_mask = torch.tensor(target_mask, dtype=torch.long).to(device)  # ++

    added_input_mask = torch.tensor(added_input_mask, dtype=torch.long).to(device)  # ++

    image = image.to(device)  # ++ #

    sentiment_label = sentiment_label.to(device).long()
    relation_label = relation_label.to(device).long()

    adj_ids = list(map(list, zip(*adj_ids)))  # ++
    adj_mask = list(map(list, zip(*adj_mask)))  # ++

    noun_ids = list(map(list, zip(*noun_ids)))  # ++
    noun_mask = list(map(list, zip(*noun_mask)))  # ++

    anp_ids = list(map(list, zip(*anp_ids)))  # ++
    anp_mask = list(map(list, zip(*anp_mask)))  # ++

    caption_ids = list(map(list, zip(*caption_ids)))  # ++
    caption_mask = list(map(list, zip(*caption_mask)))  # ++

    adj_ids = torch.tensor(adj_ids, dtype=torch.long).to(device)  # ++
    adj_mask = torch.tensor(adj_mask, dtype=torch.long).to(device)  # ++

    noun_ids = torch.tensor(noun_ids, dtype=torch.long).to(device)  # ++
    noun_mask = torch.tensor(noun_mask, dtype=torch.long).to(device)  # ++

    anp_ids = torch.tensor(anp_ids, dtype=torch.long).to(device)  # ++
    anp_mask = torch.tensor(anp_mask, dtype=torch.long).to(device)  # ++

    caption_ids = torch.tensor(caption_ids, dtype=torch.long).to(device)  # ++
    caption_mask = torch.tensor(caption_mask, dtype=torch.long).to(device)  # ++

    # GT_boxes = GT_boxes.to(device).float()
    # roi_boxes = roi_boxes.to(device).float()
    # img_feat = img_feat.to(device).float()
    # spatial_feat = spatial_feat.to(device).float()
    # box_labels = box_labels.to(device).float()

    return tokens, context_tokens, target_tokens, input_ids, context_ids, target_ids, \
           input_mask, context_mask, target_mask, added_input_mask, sentiment_label, \
           img_id, image, relation_label, adj_tokens, noun_tokens, anp_tokens, caption_tokens, \
           adj_ids, noun_ids, anp_ids, caption_ids, adj_mask, noun_mask, anp_mask, caption_mask


def main():
    start_time = datetime.datetime.now().strftime('%m-%d-%Y-%H-%M-%S_')

    # argparse is a Python module: a parser of command line options, arguments and subcommands
    # argparse.ArgumentParser() creates the parser, and ArgumentParser contains all the information needed to parse the command line into Python data types
    parser = argparse.ArgumentParser()

    # specify the command parameters that the program needs to accept
    ## Required parameters
    parser.add_argument("--dataset",
                        default='twitter2015',
                        type=str,
                        help="The name of the task to train.")  ##
    parser.add_argument("--data_dir",
                        default='./data/Sentiment_Analysis/',
                        type=str,
                        help="The input data dir. Should contain the .tsv files for the task.")
    parser.add_argument("--VG_data_dir",
                        default='./data/Image_Target_Matching',
                        type=str,
                        help="The input data dir. Should contain the .tsv files for the task.")
    parser.add_argument("--imagefeat_dir",
                        default='./twitter_images/',  # default ='./data/twitter_images/',
                        type=str,
                        )  ##
    parser.add_argument("--VG_imagefeat_dir",
                        default='/mnt/nfs-storage-titan/data/twitter_images/',  # default ='./data/twitter_images/',
                        type=str,
                        )  ##
    parser.add_argument("--output_dir",
                        default="./log/",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--save",
                        default=True,
                        action='store_true',
                        help="Whether to save model.")
    # batch size is too small and takes too much time, while the gradient oscillates badly, which is not conducive to convergence;
    # batch size is too large, the gradient direction does not change from batch to batch, and it is easy to fall into local minima.
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--SA_learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--VG_learning_rate",
                        default=1e-6,
                        type=float,
                        help="The initial learning rate for Adam.")
    # parser.add_argument("--ranking_loss_ratio",
    #                     default=0.5,
    #                     type=float)
    parser.add_argument("--pred_loss_ratio",
                        default=1.,
                        type=float)
    parser.add_argument("--num_train_epochs",
                        default=9.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for."
                             "E.g., 0.1 = 10%  of training.")
    parser.add_argument("--seed",
                        default=2020,  # 24
                        type=int,
                        help="random seed for initialization")
    # parser.add_argument("--roi_num",  # RoI is the mapping of the "candidate box" on the feature map obtained after Selective Search is completed.
    #                     default=100,
    #                     type=int)

    parser.add_argument("--reg_num",
                        default=128,
                        type=int)  # ++
    parser.add_argument("--crop_size",
                        default=224,
                        type=int,
                        help="crop size of image")  # ++  # image cropping size
    parser.add_argument("--resnet_root",
                        default="./resnet",
                        help="path the pre-trained cnn models")  # ++
    parser.add_argument("--fine_tune_cnn",
                        action="store_true",
                        help="fine tune pre-trained CNN if True")  # ++
    # parser.add_argument("--attention_parameter",
    #                     default=0.5,
    #                     type=float)  # ++  # twitter2015: 0.5  # twitter2017: 0.2
    # parser.add_argument("--anp_parameter",
    #                     default=0.6,
    #                     type=float)  # ++  # twitter2015: 0.6  # twitter2017: 0.2
    parser.add_argument("--recon_loss_ratio",
                        default=0.2,
                        type=float)  # ++  # twitter2015: 0.2  # twitter2017: 0.3
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")  # ++

    args = parser.parse_args()  # all "add_argument" set in the parser is returned to the args subclass instance.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # default is cuda:0

    args.data_dir = args.data_dir + str(args.dataset).lower() + '/%s.pkl'  # lower() to lowercase
    args.imagefeat_dir = args.imagefeat_dir + str(args.dataset).lower()
    args.VG_data_dir = args.VG_data_dir + '/%s.pkl'
    args.VG_imagefeat_dir = args.VG_imagefeat_dir + 'twitter2017'  # image-target-matching data from twitter2017
    # args.VG_imagefeat_dir = args.VG_imagefeat_dir + str(args.dataset).lower()
    args.output_dir = args.output_dir + str(args.dataset) + "/"

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))  # Exception handling

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)  # create a new directory if it doesn't exist
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    output_logger_file = os.path.join(args.output_dir, "log.txt")

    output_encoder_file = os.path.join(args.output_dir, "pytorch_encoder.bin")  # ++


    # setting the logging format
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',  # log content
                        datefmt='%m/%d/%Y %H:%M:%S',  # date format
                        level=logging.INFO,  # print log level
                        filename=output_logger_file)  # log storage location
    logger = logging.getLogger(__name__)  # initialisation log

    # write log messages
    logger.info("dataset:{}  num_train_epochs:{}".format(args.dataset, args.num_train_epochs))
    logger.info("SA_learning_rate:{}  warmup_proportion:{}".format(args.SA_learning_rate, args.warmup_proportion))
    logger.info("VG_learning_rate:{}  warmup_proportion:{}".format(args.VG_learning_rate, args.warmup_proportion))
    logger.info("pred_loss_ratio:{}  recon_loss_ratio:{}".format(args.pred_loss_ratio, args.recon_loss_ratio))
    logger.info(args)


    # generating random numbers
    random.seed(args.seed)  # the role of the random seed is to generate random numbers whose weights are initial conditions
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)



    # download pre-training model Roberta
    tokenizer = RobertaTokenizer.from_pretrained('./roberta-base')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    '''DataProcessor.py'''
    train_dataset_SA = MyDataset(args.data_dir%str('train'), args.imagefeat_dir, tokenizer, max_seq_len=args.max_seq_length, crop_size=224)  # ++
    # train_dataset_VG = MyDataset(args.VG_data_dir%str('VG_train'), args.VG_imagefeat_dir, tokenizer, max_seq_len=args.max_seq_length, crop_size=224)  # ++
    train_dataloader_SA = Data.DataLoader(dataset=train_dataset_SA, shuffle=True, batch_size=args.train_batch_size, num_workers=0)
    # train_dataloader_VG = Data.DataLoader(dataset=train_dataset_VG, shuffle=True, batch_size=args.train_batch_size, num_workers=0)

    eval_dataset_SA = MyDataset(args.data_dir%str('dev'), args.imagefeat_dir, tokenizer, max_seq_len=args.max_seq_length, crop_size=224)  # ++
    # eval_dataset_VG = MyDataset(args.VG_data_dir%str('VG_dev'), args.VG_imagefeat_dir, tokenizer, max_seq_len=args.max_seq_length, crop_size=224)  # ++
    eval_dataloader_SA = Data.DataLoader(dataset=eval_dataset_SA, shuffle=False, batch_size=args.eval_batch_size, num_workers=0)
    # eval_dataloader_VG = Data.DataLoader(dataset=eval_dataset_VG, shuffle=False, batch_size=args.eval_batch_size, num_workers=0)

    test_dataset = MyDataset(args.data_dir%str('test'), args.imagefeat_dir, tokenizer, max_seq_len=args.max_seq_length, crop_size=224)  # ++
    test_dataloader = Data.DataLoader(dataset=test_dataset, shuffle=False, batch_size=args.eval_batch_size, num_workers=0)


    # train_number = max(train_dataset_SA.number, train_dataset_VG.number)
    train_number = train_dataset_SA.number
    num_train_steps = int(train_number / args.train_batch_size * args.num_train_epochs)  # total training times, Epoch is the process of training all training samples once


    '''model.py'''
    model = Coarse2Fine(roberta_name='./roberta-base', reg_num=args.reg_num)  # ++
    # model = Coarse2Fine(roberta_name='bert-base-uncased', reg_num=args.reg_num)  # ++

    # new_state_dict = model.state_dict()
    # logger.info(new_state_dict)


    '''resnet.py'''
    net = getattr(resnet, 'resnet152')()  # ++  # getattr is used to return the value of an object attribute.
    net.load_state_dict(torch.load(os.path.join(args.resnet_root, 'resnet152.pth')))  # ++  # load resnet152
    '''resnet_utils.py'''
    encoder = myResnet(net, args.fine_tune_cnn, device)  # ++


    model.to(device)  # copy all the variables from the very first read to the GPU specified by device
    encoder.to(device)  # ++

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总量: {total_params}")

    # optimisation
    # prepare optimizer
    # optimizer BertAdam
    param_optimizer = list(model.named_parameters())  # get all parameters
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    '''
    use Warmup to warm up the learning rate, i.e., training with an initially small learning rate, then increasing it a little bit each STEP until the initially set larger learning rate is reached (note: at this point, the warm-up learning rate is complete).
    the initial learning rate is used for training (note: the learning rate is decaying during the training process after the warmup learning rate is completed), which helps to make the model converge faster.
    '''
    # formerly consisted of two optimisers, BertAdam and OpenAIAdam, now replaced by a single AdamW optimiser
    # optimizer_VG = BertAdam(optimizer_grouped_parameters,
    #                         lr=args.VG_learning_rate,
    #                         warmup=args.warmup_proportion,
    #                         t_total=num_train_steps)

    optimizer_SA = BertAdam(optimizer_grouped_parameters,
                            lr=args.SA_learning_rate,
                            warmup=args.warmup_proportion,
                            t_total=num_train_steps)

    VG_global_step = 0
    SA_global_step = 0
    nb_tr_steps = 0
    max_senti_acc = 0.0
    best_epoch = -1

    logger.info("*************** Running training ***************")
    for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):
        logger.info("************************************************** Epoch: " + str(train_idx) + " *************************************************************")
        logger.info("  Num examples = %d", train_number)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)


        ### train
        model.train()  # the role of model.train() is to enable Batch Normalisation and Dropout, model.eval() does not

        vg_l, sa_l = 0, 0
        pred_l = 0
        ranking_l, regression_l, senti_l, rel_l = 0, 0, 0, 0
        recon_l = 0  # ++
        scaler = GradScaler()
        # the enumerate function is used to iterate through the elements of a sequence and their indexes
        for batch_SA in tqdm(train_dataloader_SA, desc="Iteration"):
            '''
            #### VG
            # def post_dataloader(batch)
            # tokens, input_ids, input_mask, sentiment_label, \
            # img_id, img_shape, relation_label, GT_boxes, roi_boxes, img_feat, spatial_feat, box_labels = post_dataloader(batch_VG)

            # context_tokens、context_ids、context_mask、target_tokens、target_ids、target_mask、added_input_mask、image  # ++
            # adj_tokens、noun_tokens、adj_ids、noun_ids、adj_mask、noun_mask  # ++
            tokens, context_tokens, target_tokens, input_ids, context_ids, target_ids, \
            input_mask, context_mask, target_mask, added_input_mask, sentiment_label, \
            img_id, image, relation_label = post_dataloader_VG(batch_VG)  # ++

            with torch.no_grad():
                imgs_f, img_mean, img_att = encoder(image)  # ++

            pred_loss, pred_score = model(img_id=img_id,
                                          input_ids=input_ids,
                                          input_mask=input_mask,
                                          context_ids=context_ids,
                                          context_mask=context_mask,
                                          target_ids=target_ids,
                                          target_mask=target_mask,
                                          relation_label=relation_label,
                                          visual_sen_att=img_att,
                                          pred_loss_ratio=args.pred_loss_ratio,
                                          added_attention_mask=added_input_mask)  # ++

            # loss_VG = ranking_loss + pred_loss  # λ1*L^RE + λ2*L^ATT
            # loss_VG.backward()

            loss_VG = pred_loss  # λ1*L^RE  # ++

            # ranking_l += ranking_loss.item()
            pred_l += pred_loss.item()

            # optimisation
            lr_this_step = args.VG_learning_rate * warmup_linear(VG_global_step / num_train_steps, args.warmup_proportion)
            for param_group in optimizer_VG.param_groups:
                param_group['lr'] = lr_this_step
            optimizer_VG.step()
            optimizer_VG.zero_grad()
            VG_global_step += 1
            '''

            #### SA
            # def post_dataloader(batch)
            # SA_tokens, SA_input_ids, SA_input_mask, SA_sentiment_label, \
            # SA_img_id, SA_img_shape, SA_relation_label, SA_GT_boxes, SA_roi_boxes, SA_img_feat, SA_spatial_feat, SA_box_labels = post_dataloader(batch_SA)

            SA_tokens, SA_context_tokens, SA_target_tokens, SA_input_ids, SA_context_ids, SA_target_ids, \
            SA_input_mask, SA_context_mask, SA_target_mask, SA_added_input_mask, SA_sentiment_label, \
            SA_img_id, SA_image, SA_relation_label, SA_adj_tokens, SA_noun_tokens, SA_anp_tokens, SA_caption_tokens, \
            SA_adj_ids, SA_noun_ids, SA_anp_ids, SA_caption_ids, SA_adj_mask, SA_noun_mask, SA_anp_mask, SA_caption_mask = post_dataloader(batch_SA)  # ++

            with torch.no_grad():
                SA_imgs_f, SA_img_mean, SA_img_att = encoder(SA_image)  # ++

            with autocast():
                senti_pred = model(img_id=SA_img_id,
                                   input_ids=SA_input_ids,
                                   input_mask=SA_input_mask,
                                   context_ids=SA_context_ids,
                                   context_mask=SA_context_mask,
                                   target_ids=SA_target_ids,
                                   target_mask=SA_target_mask,
                                   relation_label=None,
                                   visual_sen_att=SA_img_att,
                                   recon_loss_ratio=args.recon_loss_ratio,
                                   adj_ids=SA_adj_ids,
                                   adj_mask=SA_adj_mask,
                                   noun_ids=SA_noun_ids,
                                   noun_mask=SA_noun_mask,
                                   anp_ids=SA_anp_ids,
                                   anp_mask=SA_anp_mask,
                                   caption_ids=SA_caption_ids,
                                   caption_mask=SA_caption_mask,
                                   added_attention_mask=SA_added_input_mask)  # ++

                senti_loss_fct = CrossEntropyLoss()  # cross entropy loss function
            # a parameter in view is set to -1, which represents a dynamic adjustment of the number of elements in this dimension to keep the total number of elements constant
                sentiment_loss = senti_loss_fct(senti_pred.view(-1, 3), SA_sentiment_label.view(-1))

                loss_SA = sentiment_loss  # L^TMSC
            # loss_SA.backward()
            scaler.scale(loss_SA).backward()
            # loss_SA = recon_loss + sentiment_loss  # λ2*L^ANP + L^TMSC  # ++

            # loss = loss_SA  # λ1*L^RE + λ2*L^ANP + L^TMSC  # ++
            # loss.backward()  # ++

            # recon_l += recon_loss.item()  # ++
            senti_l += sentiment_loss.item()  # take the elemental value of a single-element tensor and returns that value, floating-point data

            # optimisation
            lr_this_step = args.SA_learning_rate * warmup_linear(SA_global_step / num_train_steps, args.warmup_proportion)
            for param_group in optimizer_SA.param_groups:
                param_group['lr'] = lr_this_step
            # optimizer_SA.step()
            scaler.step(optimizer_SA)
            scaler.update()
            optimizer_SA.zero_grad()
            SA_global_step += 1

            nb_tr_steps += 1

        # ranking_l = ranking_l / nb_tr_steps
        pred_l = pred_l / nb_tr_steps
        # recon_l = recon_l / nb_tr_steps  # ++
        senti_l = senti_l / nb_tr_steps

        # logger.info("  ranking_loss: %s", ranking_l)
        logger.info("  pred_loss: %s", pred_l)
        # logger.info("  recon_loss: %s", recon_l)
        logger.info("  sentiment_loss: %s", senti_l)


        ### dev
        model.eval() # model.eval() does not enable Batch Normalization and Dropout

        logger.info("*****Running evaluation on Dev Set*****")
        logger.info("  SA Num examples = %d", eval_dataset_SA.number)  # len(eval_examples)
        logger.info("  Batch size = %d", args.eval_batch_size)

        nb_eval_examples = 0
        SA_nb_eval_examples = 0
        senti_acc, rel_acc = 0, 0
        ranking_vg_acc = 0
        senti_precision, senti_recall, senti_F_score = 0, 0, 0
        senti_true_label_list = []
        senti_pred_label_list = []
        num_right_vg = 0
        num_valid = 0

        '''
        #### VG
        for s, batch_VG in enumerate(tqdm(eval_dataloader_VG, desc="Evaluating_VG")):
            # def post_dataloader(batch)
            # tokens, input_ids, input_mask, sentiment_label, \
            # img_id, img_shape, relation_label, GT_boxes, roi_boxes, img_feat, spatial_feat, box_labels = post_dataloader(batch_VG)

            tokens, context_tokens, target_tokens, input_ids, context_ids, target_ids, \
            input_mask, context_mask, target_mask, added_input_mask, sentiment_label, \
            img_id, image, relation_label = post_dataloader_VG(batch_VG)  # ++

            # tell the autoderivative engine not to perform the derivation operation, which is meant to speed up the computation and save memory.
            with torch.no_grad():
                imgs_f, img_mean, img_att = encoder(image)  # ++

                pred_loss, pred_score = model(img_id=img_id,
                                              input_ids=input_ids,
                                              input_mask=input_mask,
                                              context_ids=context_ids,
                                              context_mask=context_mask,
                                              target_ids=target_ids,
                                              target_mask=target_mask,
                                              relation_label=relation_label,
                                              visual_sen_att=img_att,
                                              added_attention_mask=added_input_mask)  # ++

            current_batch_size = input_ids.size()[0]

            # -----evaluate-----
            ##### coarse-grained
            # correlation between image and target
            # the effect of detach() is to block reverse gradient propagation
            # the cpu() function works by copying data from the GPU to memory.
            pred_score = pred_score.detach().cpu().numpy()  # [N*n, 100]  # .reshape(current_batch_size, args.max_GT_boxes, -1)  # [N*n, 100] -> [N, n, 100]
            relation_pred = np.argmax(pred_score, axis=1)  # returns the index of the largest value in a numpy array.
            tmp_rel_accuracy = np.sum(relation_pred == relation_label.cpu().numpy())
            rel_acc += tmp_rel_accuracy

            # roi_boxes = roi_boxes.detach().cpu().numpy()  # [N, 100, 4]
            # GT_boxes = GT_boxes.detach().cpu()  # [N, n, 4]
            # attn_map = attn_map.detach().cpu().numpy()

            ##### fine-grained
            for i in range(current_batch_size):  # N
                if relation_label[i] != 0:
                    num_valid += 1

                    ious = (torchvision.ops.box_iou(GT_boxes[i, 0:1, :], torch.tensor(roi_boxes[i]))).numpy()  # [1, 4], [100, 4] -> [1, 100]  # If GT is 0, iou is 0
                    sorted_index = np.argsort(-attn_map[i])[0]  # fetch the minimum value of the array
                    pred_ids = sorted_index[:1]  # top K = 1
                    topk_max_iou = ious[0][pred_ids]
                    pred_iou = topk_max_iou.max()

                    if pred_iou >= 0.5:
                        num_right_vg += 1
            nb_eval_examples += current_batch_size

        rel_acc = rel_acc / nb_eval_examples
        # ranking_vg_acc = num_right_vg / num_valid
        '''

        #### SA
        for batch_SA in tqdm(eval_dataloader_SA, desc="Evaluating_SA"):
            # def post_dataloader(batch)
            # SA_tokens, SA_input_ids, SA_input_mask, SA_sentiment_label, \
            # SA_img_id, SA_img_shape, SA_relation_label, SA_GT_boxes, SA_roi_boxes, SA_img_feat, SA_spatial_feat, SA_box_labels = post_dataloader(batch_SA)

            SA_tokens, SA_context_tokens, SA_target_tokens, SA_input_ids, SA_context_ids, SA_target_ids, \
            SA_input_mask, SA_context_mask, SA_target_mask, SA_added_input_mask, SA_sentiment_label, \
            SA_img_id, SA_image, SA_relation_label, SA_adj_tokens, SA_noun_tokens, SA_anp_tokens, SA_caption_tokens, \
            SA_adj_ids, SA_noun_ids, SA_anp_ids, SA_caption_ids, SA_adj_mask, SA_noun_mask, SA_anp_mask, SA_caption_mask = post_dataloader(batch_SA)  # ++

            with torch.no_grad():
                SA_imgs_f, SA_img_mean, SA_img_att = encoder(SA_image)  # ++

                SA_senti_pred = model(img_id=SA_img_id,
                                      input_ids=SA_input_ids,
                                      input_mask=SA_input_mask,
                                      context_ids=SA_context_ids,
                                      context_mask=SA_context_mask,
                                      target_ids=SA_target_ids,
                                      target_mask=SA_target_mask,
                                      relation_label=None,
                                      visual_sen_att=SA_img_att,
                                      adj_ids=SA_adj_ids,
                                      adj_mask=SA_adj_mask,
                                      noun_ids=SA_noun_ids,
                                      noun_mask=SA_noun_mask,
                                      anp_ids=SA_anp_ids,
                                      anp_mask=SA_anp_mask,
                                      caption_ids=SA_caption_ids,
                                      caption_mask=SA_caption_mask,
                                      added_attention_mask=SA_added_input_mask)  # ++

            SA_sentiment_label = SA_sentiment_label.cpu().numpy()
            SA_senti_pred = SA_senti_pred.cpu().numpy()
            senti_true_label_list.append(SA_sentiment_label)
            senti_pred_label_list.append(SA_senti_pred)
            tmp_senti_accuracy = accuracy(SA_senti_pred, SA_sentiment_label)  # def accuracy(out, labels)
            senti_acc += tmp_senti_accuracy

            current_batch_size = SA_input_ids.size()[0]
            SA_nb_eval_examples += current_batch_size

        senti_acc = senti_acc / SA_nb_eval_examples  # predictive accuracy

        senti_true_label = np.concatenate(senti_true_label_list)  # concatenate for data splicing
        senti_pred_outputs = np.concatenate(senti_pred_label_list)
        # def macro_f1
        senti_precision, senti_recall, senti_F_score = macro_f1(senti_true_label, senti_pred_outputs)

        result = {'nb_eval_examples': nb_eval_examples,
                  # 'num_valid': num_valid,
                  'Dev_rel_acc': rel_acc,
                  'Dev_senti_acc': senti_acc,  # acc
                  'Dev_senti_precision': senti_precision,
                  'Dev_senti_recall': senti_recall,
                  'Dev_senti_F_score': senti_F_score,  # f1
                  # 'Dev_ranking_vg_acc': ranking_vg_acc,  # num_right_vg / num_valid
                  }

        logger.info("***** Dev Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))


        ### test
        model.eval()
        logger.info("***** Running evaluation on Test Set *****")
        logger.info("  Num examples = %d", test_dataset.number)
        logger.info("  Batch size = %d", args.eval_batch_size)

        nb_test_examples = 0
        test_senti_acc = 0
        test_senti_true_label_list = []
        test_senti_pred_label_list = []

        for batch in tqdm(test_dataloader, desc="Testing"):
            # def post_dataloader(batch)
            # tokens, input_ids, input_mask, sentiment_label, \
            # img_id, img_shape, relation_label, GT_boxes, roi_boxes, img_feat, spatial_feat, box_labels = post_dataloader(batch)

            tokens, context_tokens, target_tokens, input_ids, context_ids, target_ids, \
            input_mask, context_mask, target_mask, added_input_mask, sentiment_label, \
            img_id, image, relation_label, adj_tokens, noun_tokens, anp_tokens, caption_tokens, \
            adj_ids, noun_ids, anp_ids, caption_ids, adj_mask, noun_mask, anp_mask, caption_mask = post_dataloader(batch)  # ++

            with torch.no_grad():
                imgs_f, img_mean, img_att = encoder(image)  # ++

                senti_pred = model(img_id=img_id,
                                   input_ids=input_ids,
                                   input_mask=input_mask,
                                   context_ids=context_ids,
                                   context_mask=context_mask,
                                   target_ids=target_ids,
                                   target_mask=target_mask,
                                   relation_label=None,
                                   visual_sen_att=img_att,
                                   adj_ids=adj_ids,
                                   adj_mask=adj_mask,
                                   noun_ids=noun_ids,
                                   noun_mask=noun_mask,
                                   anp_ids=anp_ids,
                                   anp_mask=anp_mask,
                                   caption_ids=caption_ids,
                                   caption_mask=caption_mask,
                                   added_attention_mask=added_input_mask)  # ++

            sentiment_label = sentiment_label.cpu().numpy()
            senti_pred = senti_pred.cpu().numpy()
            test_senti_true_label_list.append(sentiment_label)
            test_senti_pred_label_list.append(senti_pred)
            tmp_senti_accuracy = accuracy(senti_pred, sentiment_label)
            test_senti_acc += tmp_senti_accuracy

            current_batch_size = input_ids.size()[0]
            nb_test_examples += current_batch_size
            
        test_senti_acc = test_senti_acc / nb_test_examples

        senti_true_label = np.concatenate(test_senti_true_label_list)
        senti_pred_outputs = np.concatenate(test_senti_pred_label_list)
        # def macro_f1
        test_senti_precision, test_senti_recall, test_senti_F_score = macro_f1(senti_true_label, senti_pred_outputs)

        result = {'Test_senti_acc': test_senti_acc,
                  'Test_senti_precision': test_senti_precision,
                  'Test_senti_recall': test_senti_recall,
                  'Test_senti_F_score': test_senti_F_score
                  }

        logger.info("***** Test Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        # save model
        if senti_acc >= max_senti_acc:
            # save a trained model
            if args.save:
                model_to_save = model.module if hasattr(model, 'module') else model
                encoder_to_save = encoder.module if hasattr(encoder, 'module') else encoder  # ++

                torch.save(model_to_save.state_dict(), output_model_file)
                torch.save(encoder_to_save.state_dict(), output_encoder_file)  # ++

            max_senti_acc = senti_acc

            corresponding_test_acc = test_senti_acc
            corresponding_test_p = test_senti_precision
            corresponding_test_r = test_senti_recall
            corresponding_test_f = test_senti_F_score

            best_epoch = train_idx

    logger.info("max_dev_senti_acc: %s ", max_senti_acc)
    logger.info("corresponding_test_sentiment_acc: %s", corresponding_test_acc)
    logger.info("corresponding_test_sentiment_precision: %s", corresponding_test_p)
    logger.info("corresponding_test_sentiment_recall: %s", corresponding_test_r)
    logger.info("corresponding_test_sentiment_F_score: %s", corresponding_test_f)
    logger.info("best_epoch: %d", best_epoch)


if __name__ == "__main__":
    main()