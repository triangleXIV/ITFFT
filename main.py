import os
import argparse
import torch
import numpy as np
from torch.optim import AdamW
from tqdm import tqdm, trange
from toolcls import seed_everything, process_tsv_file, create_dataloader, save_dataloader, load_dataloader, BertConfig
from sklearn.metrics import precision_recall_fscore_support
from logs import logger
from deberta import ITFFT
from transformers import AutoTokenizer
from torch.cuda.amp import autocast, GradScaler

def evaluate_model(eval_dataloader, output_model_file, args):
    global max_acc
    logger.info("***** Running evaluation on Dev Set*****")
    logger.info("  Num examples = %d", len(eval_dataloader.dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    true_label_list = []
    pred_label_list = []

    progress_bar = tqdm(enumerate(eval_dataloader), desc="Iteration", total=len(eval_dataloader), position=0)
    for step, batch in progress_bar:
        labels = batch["labels"].to(device)
        image_text_input_ids = batch["image_text_input_ids"].to(device)
        image_text_attention_mask = batch["image_text_attention_mask"].to(device)
        image_text_token_type_ids = batch["image_text_token_type_ids"].to(device)
        # text_entity_input_ids = batch["text_entity_input_ids"].to(device)
        # text_entity_attention_mask = batch["text_entity_attention_mask"].to(device)
        # text_entity_token_type_ids = batch["text_entity_token_type_ids"].to(device)

        with torch.no_grad():
            tmp_eval_loss,logits = model(
                                  img_text_ids=image_text_input_ids, img_text_mask=image_text_attention_mask,
                                  image_text_type_ids=image_text_token_type_ids,
                                  labels=labels)

        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        true_label_list.append(label_ids)
        pred_label_list.append(logits)
        tmp_eval_accuracy = accuracy(logits, label_ids)
        progress_bar.set_description(f"Evaluating (tmp_eval_loss: {tmp_eval_loss.item():.4f})")
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += image_text_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    loss = tr_loss / nb_tr_steps if args.do_train else None
    true_label = np.concatenate(true_label_list)
    pred_outputs = np.concatenate(pred_label_list)
    precision, recall, F_score = macro_f1(true_label, pred_outputs)
    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              'f_score': F_score,
              'global_step': global_step,
              'loss': loss}
    logger.info("***** Dev Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    if eval_accuracy >= max_acc:
        # Save a trained model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        if args.do_train:
            torch.save(model_to_save.state_dict(), output_model_file)
        max_acc = eval_accuracy

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return (1 - x) / (1 - warmup)


def macro_f1(y_true, y_pred):
    preds = np.argmax(y_pred, axis=-1)
    true = y_true
    p_macro, r_macro, f_macro, support_macro \
        = precision_recall_fscore_support(true, preds, average='macro')
    # f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return p_macro, r_macro, f_macro


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--deberta_dir", default='./pretrains', type=str,  # 文本数据位置
                        help="The pretrain model file")
    parser.add_argument("--task_name", default='twitter15', type=str,  # 要加载哪个数据集 Twitter是17 Twitter15是15
                        help="The name of the task to train.")
    parser.add_argument("--output_dir", default='./output', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--max_img_length", default=192, type=int,
                        help="Max tokens. Include image and text. ")
    parser.add_argument("--train_batch_size", default=48, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=48, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=6.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=648,
                        help="random seed for initialization")
    parser.add_argument('--bert_cfg', type=str, default="./pretrains/bert_config.json",
                        help='path to config file', )
    parser.add_argument('--do_train', type=bool, default=True,)
    parser.add_argument('--overwrite_output_dir', type=bool, default=True,
                        help='Overwriting the output directory?', )

    args = parser.parse_args(args=[])

    # 初始化输出的文件夹
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # 设置cuda
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    # 根据任务名称 定义标签 定义图片位置
    if args.task_name == "twitter17":
        args.tsv_path = "./twitter17/"
    elif args.task_name == "twitter15":
        args.tsv_path = "./twitter15/"
    else:
        print("The task name is not right!")

    seed_everything(args.seed)  # 固定随机数种子

    bert_config = BertConfig.from_json_file(args.bert_cfg)

    tokenizer = AutoTokenizer.from_pretrained(args.deberta_dir)
    train_labels, train_image_text_tokens = process_tsv_file(
        os.path.join(args.tsv_path, "new_train.tsv"), tokenizer=tokenizer,
        max_img_length=args.max_img_length,
    )
    dev_labels, dev_image_text_tokens = process_tsv_file(
        os.path.join(args.tsv_path, "new_dev.tsv"), tokenizer=tokenizer,
        max_img_length=args.max_img_length,
    )
    test_labels, test_image_text_tokens = process_tsv_file(
        os.path.join(args.tsv_path, "new_test.tsv"), tokenizer=tokenizer,
        max_img_length=args.max_img_length,
    )
    # save the tensor data, avoid to deal with the data repeatedly
    if os.path.isfile(os.path.join(args.tsv_path, "new_train.pt")):
        train_dataloader = load_dataloader(args.tsv_path, expand="new_train.pt", shuffle=True,
                                           batch_size=args.train_batch_size)
    else:
        train_dataloader = create_dataloader(train_labels, train_image_text_tokens,
                                             shuffle=True,
                                             batch_size=args.train_batch_size)
        save_dataloader(train_dataloader, args.output_dir, "train_dataloader.pt")

    if os.path.isfile(os.path.join(args.tsv_path, "new_dev.pt")):
        dev_dataloader = load_dataloader(args.tsv_path, expand="new_dev.pt", shuffle=False,
                                         batch_size=args.dev_batch_size)
    else:
        dev_dataloader = create_dataloader(dev_labels, dev_image_text_tokens, shuffle=False,
                                           batch_size=args.train_batch_size)
        save_dataloader(dev_dataloader, args.output_dir, "dev_dataloader.pt")

    if os.path.isfile(os.path.join(args.tsv_path, "new_test.pt")):
        test_dataloader = load_dataloader(args.tsv_path, expand="new_test.pt", shuffle=False,
                                          batch_size=args.eval_batch_size)
    else:
        test_dataloader = create_dataloader(test_labels, test_image_text_tokens, shuffle=False,
                                            batch_size=args.eval_batch_size)
        save_dataloader(test_dataloader, args.output_dir, "test_dataloader.pt")

    output_model_file = os.path.join(args.output_dir, "pytorch_model.pth")

    avg_acc = 0.0
    avg_f1 = 0.0

    for i in range(10):
        # 准备训练的参数
        train_examples = len(train_dataloader.dataset)
        num_train_steps = int(train_examples / args.train_batch_size * args.num_train_epochs)
        t_total = num_train_steps
        torch.cuda.empty_cache()  # 先把cuda清空了
        model = ITFFT(args, bert_config)

        total_params = sum(p.numel() for p in model.parameters())
        print(f'Total number of parameters: {total_params}')

        model.to(device)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]


        def check_keywords_in_name(name, keywords=()):
            isin = False
            for keyword in keywords:
                if keyword in name:
                    isin = True
            return isin


        def set_weight_decay(model, skip_list=(), skip_keywords=()):
            has_decay = []
            no_decay = []

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                        check_keywords_in_name(name, skip_keywords):
                    no_decay.append(param)
                else:
                    has_decay.append(param)
            return [{'params': has_decay, 'weight_decay': 0.01},
                    {'params': no_decay, 'weight_decay': 0.}]


        skip = {'absolute_pos_embed'}
        skip_keywords = {'relative_position_bias_table'}
        optimizer_params = set_weight_decay(model, skip_list=skip, skip_keywords=skip_keywords)
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-6)

        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
        max_acc = 0.0
        # '''
        scaler = GradScaler()
        logger.info("*************** Running training ***************")
        for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):
            logger.info("********** Epoch: " + str(train_idx) + " **********")
            logger.info("  Num examples = %d", train_examples)
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_steps)
            model.train()

            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            progress_bar = tqdm(enumerate(train_dataloader), desc="Iteration", total=len(train_dataloader), position=0)
            for step, batch in progress_bar:
                labels = batch["labels"].to(device)
                image_text_input_ids = batch["image_text_input_ids"].to(device)
                image_text_attention_mask = batch["image_text_attention_mask"].to(device)
                image_text_token_type_ids = batch["image_text_token_type_ids"].to(device)
                # text_entity_input_ids = batch["text_entity_input_ids"].to(device)
                # text_entity_attention_mask = batch["text_entity_attention_mask"].to(device)
                # text_entity_token_type_ids = batch["text_entity_token_type_ids"].to(device)

                with autocast():
                    loss,_ = model(
                        img_text_ids=image_text_input_ids, img_text_mask=image_text_attention_mask,
                        image_text_type_ids=image_text_token_type_ids,
                        labels=labels)

                scaler.scale(loss).backward()

                tr_loss += loss.detach().item()  # 记录未缩放的损失值
                nb_tr_examples += image_text_input_ids.size(0)
                nb_tr_steps += 1

                lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                    progress_bar.set_description(f"Iteration (loss: {loss.item():.4f},lr:{param_group['lr']:.10f})")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                global_step += 1

                if (global_step % 80 == 0):
                    evaluate_model(dev_dataloader, output_model_file, args)
                    evaluate_model(test_dataloader, output_model_file, args)

        evaluate_model(dev_dataloader, output_model_file, args)

        torch.cuda.empty_cache()  # 先把cuda清空了 最终的test验证
        model.load_state_dict(torch.load(output_model_file))
        logger.info("***** Running evaluation on Test Set*****")
        logger.info("  Num examples = %d", len(test_dataloader.dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        true_label_list = []
        pred_label_list = []

        progress_bar = tqdm(enumerate(test_dataloader), desc="Iteration", total=len(test_dataloader),
                            position=0)
        for step, batch in progress_bar:
            labels = batch["labels"].to(device)
            image_text_input_ids = batch["image_text_input_ids"].to(device)
            image_text_attention_mask = batch["image_text_attention_mask"].to(device)
            image_text_token_type_ids = batch["image_text_token_type_ids"].to(device)
            # text_entity_input_ids = batch["text_entity_input_ids"].to(device)
            # text_entity_attention_mask = batch["text_entity_attention_mask"].to(device)
            # text_entity_token_type_ids = batch["text_entity_token_type_ids"].to(device)
            with torch.no_grad():
                tmp_eval_loss,logits = model(
                    img_text_ids=image_text_input_ids, img_text_mask=image_text_attention_mask,
                    image_text_type_ids=image_text_token_type_ids,
                    labels=labels)

            logits = logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()
            true_label_list.append(label_ids)
            pred_label_list.append(logits)
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += image_text_input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        loss = tr_loss / nb_tr_steps if args.do_train else None
        true_label = np.concatenate(true_label_list)
        pred_outputs = np.concatenate(pred_label_list)
        precision, recall, F_score = macro_f1(true_label, pred_outputs)
        avg_acc += eval_accuracy
        avg_f1 += F_score
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'precision': precision,
                  'recall': recall,
                  'f_score': F_score,
                  'global_step': global_step,
                  'loss': loss}

        pred_label = np.argmax(pred_outputs, axis=-1)
        fout_p = open(os.path.join(args.output_dir, "pred.txt"), 'w')
        fout_t = open(os.path.join(args.output_dir, "true.txt"), 'w')

        for i in range(len(pred_label)):
            attstr = str(pred_label[i])
            fout_p.write(attstr + '\n')
        for i in range(len(true_label)):
            attstr = str(true_label[i])
            fout_t.write(attstr + '\n')
        print(result)

    print("avg_acc: " + str(avg_acc / 10.0))
    print("avg_f1: " + str(avg_f1 / 10.0))
