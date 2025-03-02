import copy
import math
import sys

import torch
from torch import nn, reshape
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

'''modeling_utils.py'''
from modeling_utils import MultimodalityFusionLayer, BertCrossEncoder, BertLayerNorm, BertPooler
import torch.nn.functional as F

from transformers import RobertaModel, AutoConfig
from transformers import (
    BertModel,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)  # ++

import logging
logger = logging.getLogger(__name__)


class Coarse2Fine(nn.Module):
    # img_feat_dim=2048：the image feature dimension is 2048
    # roi_num=100：take the first 100 object proposals
    def __init__(self, roberta_name='roberta-base', reg_num=49):
        super().__init__()
        # self.img_feat_dim = img_feat_dim
        config = AutoConfig.from_pretrained(roberta_name)
        self.hidden_dim = config.hidden_size

        self.roberta = RobertaModel.from_pretrained(roberta_name)
        # self.sent_dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.feat_linear = nn.Linear(self.img_feat_dim, self.hidden_dim)
        # self.img_self_attn = MultimodalityFusionLayer(config, layer_num=1)

        self.vismap2sen = nn.Linear(2048, self.hidden_dim)  # ++
        self.sen2img_attention = BertCrossEncoder(config, layer_num=1)  # ++
        self.img2sen_attention = BertCrossEncoder(config, layer_num=1)  # ++
        self.sen2sen_attention = BertCrossEncoder(config, layer_num=1)  # ++
        self.tar2img_attention = BertCrossEncoder(config, layer_num=1)  # ++
        self.tar2con_attention = BertCrossEncoder(config, layer_num=1)  # ++
        self.con2img_attention = BertCrossEncoder(config, layer_num=1)  # ++
        self.img2con_attention = BertCrossEncoder(config, layer_num=1)  # ++
        self.txt2anp_attention = BertCrossEncoder(config, layer_num=1)  # ++

        # self.v2t = BertCrossEncoder_AttnMap(config, layer_num=1)
        # self.t2v = BertCrossEncoder_AttnMap(config, layer_num=1)
        
        self.dropout1 = nn.Dropout(0.3)  # each neuron has a 0.3 chance of not being activated
        self.gather = nn.Linear(self.hidden_dim * 2, 1)
        self.dropout2 = nn.Dropout(0.3)
        # self.pred = nn.Linear(roi_num, 2)
        self.pred = nn.Linear(reg_num, 2)  # ++
        self.ce_loss = nn.CrossEntropyLoss()

        # self.ranking_loss = nn.KLDivLoss(reduction='batchmean')  # KL dispersion
        self.recon_loss = nn.KLDivLoss(reduction='batchmean')  # ++

        self.gate = nn.Linear(self.hidden_dim, self.hidden_dim)  # ++

        # self.gate = nn.Linear(self.hidden_dim * 2, self.hidden_dim)  # ++
        self.line = nn.Linear(self.hidden_dim * 2, self.hidden_dim)  # ++

        self.senti_self_attn = MultimodalityFusionLayer(config,layernum=1)
        # self.tar2img_pooler = BertPooler(config)  # ++
        self.first_pooler = BertPooler(config)  # first pooler
        self.senti_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.senti_detc = nn.Linear(self.hidden_dim, 3)  # triple classification
        
        self.init_weight()

    # Bert initialisation
    def init_weight(self):
        for name, module in self.named_modules():
            # isinstance() determines whether an object is of a known type
            if isinstance(module, (nn.Linear, nn.Embedding)) and ('roberta' not in name):  # determine if the module is linear/embedded
                module.weight.data.normal_(mean=0.0, std=0.02)  # the mean of a normal distribution is mean, and the standard deviation is std
            elif isinstance(module, BertLayerNorm) and ('roberta' not in name):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None and ('roberta' not in name):
                module.bias.data.zero_()


    # Define forward propagation
    def forward(self, img_id,
                input_ids, input_mask,
                context_ids, context_mask,  # ++
                target_ids, target_mask,  # ++
                relation_label, visual_sen_att,  # ++
                pred_loss_ratio=1., recon_loss_ratio=1.,  # ++
                adj_ids=None, adj_mask=None,  # ++
                noun_ids=None, noun_mask=None,  # ++
                anp_ids=None, anp_mask=None,  # ++
                caption_ids=None, caption_mask=None,  # ++
                added_attention_mask=None):  # ++

        # input_ids, input_mask : [N, L]
        #              img_feat : [N, 100, 2048]
        #          spatial_feat : [N, 100, 5]
        #             box_label : [N, 1, 100]
        # box_labels : if IoU > 0.5, IoU_i / sum(); 0
        # ranking_loss_ratio, pred_loss_ratio : λ2, λ1

        # visual_embeds_att : myResnet encoder  # ++
        # added_attention_mask : added_input_mask  # ++


        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  # Default to cuda: 0
        batch_size, seq = input_ids.size()
        # _, roi_num, feat_dim = img_feat.size()  # = 100


        # text feature
        # sentence
        roberta_output = self.roberta(input_ids, input_mask)
        sentence_output = roberta_output.last_hidden_state
        text_pooled_output = roberta_output.pooler_output  # max pooling to obtain the most significant features of relevant classifications

        # sentence_output = self.sent_dropout(sentence_output)

        # context
        context_roberta_output = self.roberta(context_ids, context_mask)  # ++
        context_output = context_roberta_output.last_hidden_state  # ++

        # target
        target_roberta_output = self.roberta(target_ids, target_mask)  # ++
        target_output = target_roberta_output.last_hidden_state  # ++


        '''
        # visual self attention
        img_feat_ = self.feat_linear(img_feat)  # [N*n, 100, 2048] -> [N*n, 100, 768]
        image_mask = torch.ones((batch_size, roi_num)).to(device)  # Return a tensor that is all 1
        extended_image_mask = image_mask.unsqueeze(1).unsqueeze(2)
        extended_image_mask = extended_image_mask.to(dtype=next(self.parameters()).dtype)
        extended_image_mask = (1.0 - extended_image_mask) * -10000.0
        visual_output = self.img_self_attn(img_feat_, extended_image_mask)  # image self attention
        visual_output = visual_output[-1]  # [N*n, 100, 768]
        '''


        # ++  # apply target-based attention mechanism to obtain different image representations
        img_mask = added_attention_mask[:, :49]  # ++
        extended_img_mask = img_mask.unsqueeze(1).unsqueeze(2)
        extended_img_mask = extended_img_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_img_mask = (1.0 - extended_img_mask) * -10000.0

        vis_sen_map = visual_sen_att.view(-1, 2048, 49).permute(0, 2, 1)  # ++  # self.batch_size, 49, 2048
        converted_vis_sen_map = self.vismap2sen(vis_sen_map)  # ++  # self.batch_size, 49, hidden_dim


        # ++  # sentence representations
        extended_sent_mask = input_mask.unsqueeze(1).unsqueeze(2)
        extended_sent_mask = extended_sent_mask.to(dtype=next(self.parameters()).dtype)
        extended_sent_mask = (1.0 - extended_sent_mask) * -10000.0


        # ++  # context representations
        extended_con_mask = context_mask.unsqueeze(1).unsqueeze(2)
        extended_con_mask = extended_con_mask.to(dtype=next(self.parameters()).dtype)
        extended_con_mask = (1.0 - extended_con_mask) * -10000.0


        # ++  # target representations
        extended_tar_mask = target_mask.unsqueeze(1).unsqueeze(2)
        extended_tar_mask = extended_tar_mask.to(dtype=next(self.parameters()).dtype)
        extended_tar_mask = (1.0 - extended_tar_mask) * -10000.0


        # ++  # 1. coarse-grained matching: sentence query visual
        image_aware_sentence = self.sen2img_attention(sentence_output,
                                                         converted_vis_sen_map,
                                                         extended_img_mask)  # sentence query image
        image_aware_sentence = image_aware_sentence[-1]

        sentence_aware_image = self.img2sen_attention(converted_vis_sen_map,
                                                         sentence_output,
                                                         extended_sent_mask)  # image query sentence
        sentence_aware_image = sentence_aware_image[-1]

        sentence_aware_sentence = self.sen2sen_attention(sentence_output,
                                                            sentence_aware_image,
                                                            extended_img_mask)  # image query sentence
        sentence_aware_sentence = sentence_aware_sentence[-1]

        merge_representation = torch.cat((sentence_aware_sentence, image_aware_sentence), dim=-1)
        # gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, text_len, hidden_dim
        # gated_image_aware_sentence = gate_value * image_aware_sentence

        gate_merge_representation = torch.softmax(self.line(merge_representation), dim=-1)  # g
        gate_sentence_representation = torch.neg(gate_merge_representation).add(1)  # 1-g

        # gathered_merge_representation = self.gather(self.dropout1(merge_representation)).squeeze(2)  # [N, 128, 1536] -> [N, 128, 1] -> [N, 128]
        # rel_pred = self.pred(self.dropout2(gathered_merge_representation))  # [N, 2]

        '''
        # ++  # relevance supervision
        if relation_label != None:
            pred_loss = self.ce_loss(rel_pred, relation_label.long())  # cross entropy loss
        else:
            pred_loss = torch.tensor(0., requires_grad=True).to(device)
        '''


        '''
        # 1. visual query sentence
        extended_sent_mask = input_mask.unsqueeze(1).unsqueeze(2)
        extended_sent_mask = extended_sent_mask.to(dtype=next(self.parameters()).dtype)
        extended_sent_mask = (1.0 - extended_sent_mask) * -10000.0
        sentence_aware_image, _ = self.v2t(visual_output,
                                           sentence_output,
                                           extended_sent_mask,
                                           output_all_encoded_layers=False)  # image query sentence
        sentence_aware_image = sentence_aware_image[-1]  # [N, 100, 768]
        
        gathered_sentence_aware_image = self.gather(self.dropout1(
                                                    sentence_aware_image)).squeeze(2)  # [N, 100, 768] -> [N, 100, 1] -> [N, 100]
        rel_pred = self.pred(self.dropout2(gathered_sentence_aware_image))  # [N, 2]
        
        gate = torch.softmax(rel_pred, dim=-1)[:, 1].unsqueeze(1).expand(batch_size,
                                                                         roi_num).unsqueeze(2).expand(batch_size,
                                                                                                      roi_num, self.hidden_dim)
        gated_sentence_aware_image = gate * sentence_aware_image
        '''


        # ++  # 2. fine-grained matching
        if adj_ids != None and noun_ids != None and caption_ids != None:
            # anp: adj
            adj_roberta_output = self.roberta(adj_ids, adj_mask)  # ++
            adj_output = adj_roberta_output.last_hidden_state  # ++

            # anp: noun
            noun_roberta_output = self.roberta(noun_ids, noun_mask)  # ++
            noun_output = noun_roberta_output.last_hidden_state  # ++

            # ++  # anp
            anp_roberta_output = self.roberta(anp_ids, anp_mask)  # ++
            anp_output = anp_roberta_output.last_hidden_state  # ++

            extended_anp_mask = anp_mask.unsqueeze(1).unsqueeze(2)
            extended_anp_mask = extended_anp_mask.to(dtype=next(self.parameters()).dtype)
            extended_anp_mask = (1.0 - extended_anp_mask) * -10000.0

            # caption
            caption_roberta_output = self.roberta(caption_ids, caption_mask)  # ++
            caption_output = caption_roberta_output.last_hidden_state  # ++

            # caption representations
            extended_cap_mask = caption_mask.unsqueeze(1).unsqueeze(2)
            extended_cap_mask = extended_cap_mask.to(dtype=next(self.parameters()).dtype)
            extended_cap_mask = (1.0 - extended_cap_mask) * -10000.0

            # ++  # ANPs
            anps_aware_sentence= self.txt2anp_attention(sentence_output,
                                                            anp_output,
                                                            extended_anp_mask)  # 128

            anps_aware_sentence = anps_aware_sentence[-1]

            image_output = self.line(torch.cat((image_aware_sentence, torch.mul(0.8, anps_aware_sentence)), dim=-1))

            sentence_representation_output = torch.mul(gate_sentence_representation, sentence_output)  # (1-g)*(Hs;α*Hgcn)
            image_aware_sentence_output = torch.mul(gate_merge_representation, image_output)  # g*(Hs→v;β*Hs→a)

            weighted_output = torch.add(sentence_representation_output, image_aware_sentence_output)  # (1-g)*Hs + g*Hs→v

            '''
            # anp
            # nouns
            # modulo |HT|、|HN|
            # torch.sum() sums up a certain dimension of the input tensor data
            target_norm_output = torch.sqrt(torch.sum(torch.square(target_output), dim=2))  # HT: target  # e.g. Jake Paul
            anp_norm_output = torch.sqrt(torch.sum(torch.square(noun_output), dim=2))  # HN: noun  # e.g. crowd lights crowd festival concert

            # inner product
            # torch.mul() is the multiplication of the corresponding elements of a matrix
            sen_pivot_output = torch.sum(torch.mul(target_output, noun_output), dim=2)

            # calculate cosine similarity
            cosin = torch.div(sen_pivot_output, torch.mul(target_norm_output, anp_norm_output))  # cos(HT, HN)
            weight_ = torch.unsqueeze(cosin, -1)  # torch.unsuqueue() is used to extend the dimension, -1 represents the last dimension ##
            # recon_target_output = weight_ * noun_output  # H~N = α^m * HN^m noun
            anp_target_output = weight_ * anp_output  # ++  # H~ANP = α^m * HANP^m Adjective noun pairs

            gate_anp = torch.softmax(rel_pred, dim=-1)[:, 1].unsqueeze(1).expand(
                                     batch_size, 32).unsqueeze(2).expand(batch_size, 32, self.hidden_dim)  # ++

            recon_target_output = gate_anp * anp_target_output  # ++  # H′ANP = G ⊙ HANP

            # anp_target_output = target_output + attention_parameter * recon_target_output  # HT = HT + λN * H~N  # λN is a hyperparameter

            ## s1: target-image
            # gate_img = torch.softmax(rel_pred, dim=-1)[:, 1].unsqueeze(1).expand(
            #                          batch_size, 128).unsqueeze(2).expand(batch_size, 128, self.hidden_dim)  # ++

            # gated_image_aware_sentence = gate_img * image_aware_sentence  # H′′V = G ⊙ H′V

            gate_img = torch.softmax(rel_pred, dim=-1)[:, 1].unsqueeze(1).expand(
                                     batch_size, 128).unsqueeze(2).expand(batch_size, 128, self.hidden_dim)  # ++

            gated_image_aware_sentence = gate_img * image_aware_sentence  # ++

            s1_cross_encoder, _ = self.tar2img_attention(target_output,
                                                         gated_image_aware_sentence,
                                                         extended_sent_mask)  ##
            s1_cross_output_layer = s1_cross_encoder[-1]  ##
            '''

            '''
            # adjectives
            query_adj_output = weight_ * adj_output  # H~A = α^m * HA^m  # adjective corresponding to the noun
            s1_cross_output_layer = s1_cross_output_layer + anp_parameter * query_adj_output  # HT→V = HT→V + λA * H~A  # λA is a hyperparameter
            '''

            # s1_cross_output = self.tar2img_pooler(s1_cross_output_layer)  ##
            # transpose_img_embed = s1_cross_output.unsqueeze(1)  ##

            '''
            # 2. cls query gated_visual: ranking
            image_aware_sentence, Attn_map = self.t2v(text_pooled_output.unsqueeze(1),
                                                      gated_sentence_aware_image,
                                                      extended_image_mask)  # [N, 1, 768]
            image_aware_sentence = image_aware_sentence[-1]  # [N, 1, 768]
            Attn_map = Attn_map[-1]  # [N, 1, 100]
            '''

            '''
            ## s2: target-context
            s2_cross_encoder, _ = self.tar2con_attention(target_output,
                                                         context_output,
                                                         extended_con_mask)  ##
            s2_cross_output_layer = s2_cross_encoder[-1]  ##
            '''

            '''
            ## s3: target-caption
            caption_aware_sentence, _ = self.sen2cap_attention(sentence_output,
                                                               caption_output,
                                                               extended_cap_mask,
                                                               output_all_encoded_layers=False)  # sentence query caption
            caption_aware_sentence = caption_aware_sentence[-1]

            gate_cap = torch.softmax(rel_pred, dim=-1)[:, 1].unsqueeze(1).expand(
                                     batch_size, 128).unsqueeze(2).expand(batch_size, 128, self.hidden_dim)  # ++

            gated_caption_aware_sentence = gate_cap * caption_aware_sentence  # ++  # H′CAP = G ⊙ HCAP

            s3_cross_encoder, _ = self.tar2cap_attention(target_output,
                                                         gated_caption_aware_sentence,
                                                         extended_vg_mask)  ##
            s3_cross_output_layer = s3_cross_encoder[-1]  ##
            '''

            '''
            # ++  # recon loss
            reconstruction_loss = torch.square(recon_target_output - s1_cross_output_layer)
            recon_loss_ = torch.mean(torch.mean(reconstruction_loss, 1), -1)
            recon_loss = torch.mean(recon_loss_)
            '''

            '''
            # ranking loss: kl_div
            if relation_label != None and box_labels != None:
                box_labels = box_labels.reshape(-1, roi_num)  # [N*n, 100]
                pred_loss = self.ce_loss(rel_pred, relation_label.long())
                ranking_loss = self.ranking_loss(F.softmax(Attn_map.squeeze(1), dim=1).log(), box_labels)  # box_label: soft label  # 0.7927
            else:
                pred_loss = torch.tensor(0., requires_grad=True).to(device)
                ranking_loss = torch.tensor(0., requires_grad=True).to(device)
            '''

            '''
            # ++  # sentiment classifier
            s1_final_cross_encoder, _ = self.con2img_attention(s2_cross_output_layer,
                                                               s1_cross_output_layer,
                                                               extended_tar_mask)  ##
            s1_final_cross_output_layer = s1_final_cross_encoder[-1]  ##
            '''

            '''
            s2_final_cross_encoder, _ = self.img2con_attention(s1_cross_output_layer,
                                                               s2_cross_output_layer,
                                                               extended_tar_mask)  ##
            s2_final_cross_output_layer = s2_final_cross_encoder[-1]  ##

            gate_context_image = torch.sigmoid(self.gate(s1_final_cross_output))
            s1_final_cross_output_layer = gate_context_image * s1_final_cross_output

            gate_image_context = 1 - gate_context_image
            s2_final_cross_output_layer = gate_image_context * s2_cross_output_layer
            '''

            '''
            senti_mixed_feature = torch.cat((s1_final_cross_output_layer, s2_cross_output_layer), dim=1)  # concatenate
            senti_mask = torch.ones((batch_size, 1)).to(device)
            senti_mask = torch.cat((target_mask, target_mask), dim=-1).to(device)  ##

            extended_senti_mask = senti_mask.unsqueeze(1).unsqueeze(2)
            extended_senti_mask = extended_senti_mask.to(dtype=next(self.parameters()).dtype)
            extended_senti_mask = (1.0 - extended_senti_mask) * -10000.0

            senti_mixed_output = self.senti_self_attn(senti_mixed_feature, extended_senti_mask)
            senti_mixed_output = senti_mixed_output[-1]

            senti_comb_img_output = self.first_pooler(senti_mixed_output)
            senti_pooled_output = self.senti_dropout(senti_comb_img_output)
            '''

            senti_comb_img_output = self.first_pooler(weighted_output)
            # senti_pooled_output = self.senti_dropout(senti_comb_img_output)
            senti_pred = self.senti_detc(senti_comb_img_output)  # classify

            '''
            # ++  # sentiment classifier
            # context_image_feature = torch.cat((s2_cross_output_layer, s1_cross_output_layer), dim=-1)
            context_image_feature = s2_cross_output_layer + s1_cross_output_layer
            gate_context_image = torch.sigmoid(self.gate(context_image_feature))
            s1_final_cross_output_layer = torch.mul(gate_context_image, s1_cross_output_layer)
            s2_final_cross_output = 1 - gate_context_image
            s2_final_cross_output_layer = torch.mul(s2_final_cross_output, s2_cross_output_layer)

            # context_caption_feature = torch.cat((s2_cross_output_layer, s3_cross_output_layer), dim=-1)
            # context_caption_feature = s2_cross_output_layer + s3_cross_output_layer
            # gate_context_caption = torch.sigmoid(self.gate(context_caption_feature))
            # s3_final_cross_output_layer = torch.mul(gate_context_caption, s3_cross_output_layer)

            # senti_mixed_output = torch.cat((s2_cross_output_layer, s1_final_cross_output_layer, s3_final_cross_output_layer), dim=-1)
            senti_mixed_output = s1_final_cross_output_layer + s2_final_cross_output_layer

            senti_comb_img_output = self.first_pooler(senti_mixed_output)
            senti_pooled_output = self.senti_dropout(senti_comb_img_output)
            senti_pred = self.senti_detc(senti_pooled_output)  # classify
            '''

            '''
            # ++  # sentiment classifier
            s1_final_cross_encoder, _ = self.con2img_attention(s2_cross_output_layer,
                                                               s1_cross_output_layer,
                                                               extended_tar_mask)  ##
            s1_final_cross_output_layer = s1_final_cross_encoder[-1]  ##

            s3_final_cross_encoder, _ = self.tar2cap_attention(s2_cross_output_layer,
                                                               s3_cross_output_layer,
                                                               extended_tar_mask)  ##
            s3_final_cross_output_layer = s3_final_cross_encoder[-1]  ##

            # context_caption_feature = s2_cross_output_layer + s3_cross_output_layer
            # gate_context_caption = torch.sigmoid(self.gate(context_caption_feature))
            # s3_final_cross_output_layer = torch.mul(gate_context_caption, s3_cross_output_layer)
            '''

            '''
            # sentiment classifier
            senti_mixed_feature = torch.cat((image_aware_sentence, sentence_output), dim=1)  # [N, 1+L, 768]
            senti_mask = torch.ones((batch_size, 1)).to(device)
            senti_mask = torch.cat((senti_mask, input_mask), dim=-1).to(device)
            extended_senti_mask = senti_mask.unsqueeze(1).unsqueeze(2)
            extended_senti_mask = extended_senti_mask.to(dtype=next(self.parameters()).dtype)
            extended_senti_mask = (1.0 - extended_senti_mask) * -10000.0
            senti_mixed_output = self.senti_self_attn(senti_mixed_feature, extended_senti_mask)  # [N, L+1, 768]
            senti_mixed_output = senti_mixed_output[-1]

            senti_comb_img_output = self.first_pooler(senti_mixed_output)
            senti_pooled_output = self.senti_dropout(senti_comb_img_output)
            senti_pred = self.senti_detc(senti_pooled_output)
            '''


            # affective forecasting：senti_pred
            # KL divergence：ranking_loss_ratio*ranking_loss  # --
            # cross entropy loss：pred_loss_ratio*pred_loss
            # KL divergence：recon_loss_ratio*recon_loss  # ++
            # correlation prediction：rel_pred
            # Attention Map：Attn_map  # --

            return senti_pred  # ++
        # else:
        #     return pred_loss_ratio * pred_loss, rel_pred  # ++