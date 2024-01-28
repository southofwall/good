import torch
from torch import nn

import torch.nn.functional as F
from .modeling_IFA import IFAModel
from deepke.relation_extraction.multimodal.hzk_models.modeling_hzk import HZK
from transformers import BertConfig, BertModel, CLIPConfig, CLIPModel

class IFAREModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(IFAREModel, self).__init__()

        self.args = args

        # zs
        self.vision_config = CLIPConfig.from_pretrained(self.args.vit_name).vision_config
        self.text_config = BertConfig.from_pretrained(self.args.bert_name)

        clip_model_dict = CLIPModel.from_pretrained(self.args.vit_name).vision_model.state_dict()
        bert_model_dict = BertModel.from_pretrained(self.args.bert_name).state_dict()

        print(self.vision_config)
        print(self.text_config)

        # for re
        self.vision_config.device = args.device
        self.model = IFAModel(self.vision_config, self.text_config,self.args)


        # load:
        vision_names, text_names = [], []
        model_dict = self.model.state_dict()

        # zs
        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
            elif 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '')
                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = bert_model_dict[text_name]
        assert len(vision_names) == len(clip_model_dict) and len(text_names) == len(bert_model_dict), \
                    (len(vision_names), len(text_names), len(clip_model_dict), len(bert_model_dict))
        self.model.load_state_dict(model_dict)

        # zs
        self.model.resize_token_embeddings(len(tokenizer))
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer
        # self.classifier = nn.Linear(self.text_config.hidden_size*2, num_labels)
        self.classifier = nn.Linear(self.text_config.hidden_size, num_labels)

        self.dropout = nn.Dropout(0.5)
        # self.model = HZK(args)

    def forward(
            self, 
            # [32,80]
            input_ids=None,
            # [32,80]
            attention_mask=None,
            # [32,80]
            token_type_ids=None,
            labels=None, 
            images=None, 
            aux_imgs=None,
            rcnn_imgs=None,
    ):
        #批次32
        bsz = input_ids.size(0)
        # zs IFA
        # (32,61)(32,61)(32.61)(32,3,224,224)(32,3,3,224,224)(32,3,3,224,224)
        output,loss,output_v= self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,

                            pixel_values=images,
                            aux_values=aux_imgs,
                            rcnn_values=rcnn_imgs,
                            return_dict=True,)


        # [32,80,768],[32,768]
        last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
        bsz, seq_len, hidden_size = last_hidden_state.shape

        # entity_hidden_state = torch.Tensor(bsz, 2*hidden_size) # batch, 2*hidden
        entity_hidden_state = torch.Tensor(bsz, hidden_size) # batch, 2*hidden

        # HSN
        hzk_hidden_matrix=torch.Tensor(hidden_size,hidden_size)
        hzk_hidden_matrix_t=torch.Tensor(hidden_size,hidden_size)

        for i in range(bsz):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            # [768]
            head_hidden = last_hidden_state[i, head_idx, :].squeeze()
            tail_hidden = last_hidden_state[i, tail_idx, :].squeeze()

            # hzk
            hzk_hidden_matrix=torch.matmul(head_hidden.reshape(hidden_size,1),tail_hidden.reshape(1,hidden_size))
            x = (torch.mean(hzk_hidden_matrix, dim=1, keepdim=True))
            y = (torch.mean(hzk_hidden_matrix, dim=0, keepdim=True))

            # hzk2
            # hzk_hidden_matrix=torch.matmul(head_hidden.reshape(hidden_size,1),head_hidden.reshape(1,hidden_size))
            # x_h = (torch.mean(hzk_hidden_matrix, dim=1, keepdim=True))
            # y_h = (torch.mean(hzk_hidden_matrix, dim=0, keepdim=True))
            # hzk_hidden_matrix_t=torch.matmul(tail_hidden.reshape(hidden_size,1),tail_hidden.reshape(1,hidden_size))
            # x_t = (torch.mean(hzk_hidden_matrix, dim=1, keepdim=True))
            # y_t = (torch.mean(hzk_hidden_matrix, dim=0, keepdim=True))




                # [1536]0
            # entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)

            # hzk1（768，class，entity）
            entity_hidden_state[i] = (torch.add(torch.matmul(hzk_hidden_matrix,x),torch.matmul(hzk_hidden_matrix,y.T))/2).view(-1)
            # print(entity_hidden_state)

        #     hzk2
        #     entity_hidden_state[i] = torch.cat([(torch.add(torch.matmul(hzk_hidden_matrix,x_h),torch.matmul(hzk_hidden_matrix,y_h.T))/2).view(-1), (torch.add(torch.matmul(hzk_hidden_matrix_t,x_t),torch.matmul(hzk_hidden_matrix_t,y_t.T))/2).view(-1),], dim=-1)

        # (32,1536)
        entity_hidden_state = entity_hidden_state.to(self.args.device)
        # (32,23)
        logits = self.classifier(entity_hidden_state)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            # loss
            return loss+loss_fn(logits, labels.view(-1)), logits
            # return loss_fn(logits, labels.view(-1)), logits
        return logits

class IFAREModel_base(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(IFAREModel_base, self).__init__()

        self.args = args

        # zs
        self.vision_config = CLIPConfig.from_pretrained(self.args.vit_name).vision_config
        self.text_config = BertConfig.from_pretrained(self.args.bert_name)

        clip_model_dict = CLIPModel.from_pretrained(self.args.vit_name).vision_model.state_dict()
        bert_model_dict = BertModel.from_pretrained(self.args.bert_name).state_dict()

        print(self.vision_config)
        print(self.text_config)

        # for re
        self.vision_config.device = args.device
        self.model = IFAModel(self.vision_config, self.text_config, self.args)

        # load:
        vision_names, text_names = [], []
        model_dict = self.model.state_dict()

        # zs
        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
            elif 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '')
                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = bert_model_dict[text_name]
        assert len(vision_names) == len(clip_model_dict) and len(text_names) == len(bert_model_dict), \
            (len(vision_names), len(text_names), len(clip_model_dict), len(bert_model_dict))
        self.model.load_state_dict(model_dict)

        # zs
        self.model.resize_token_embeddings(len(tokenizer))
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer
        self.classifier = nn.Linear(self.text_config.hidden_size*2, num_labels)
        # self.classifier = nn.Linear(self.text_config.hidden_size, num_labels)

        self.dropout = nn.Dropout(0.5)
        # self.model = HZK(args)

    def forward(
            self,
            # [32,80]
            input_ids=None,
            # [32,80]
            attention_mask=None,
            # [32,80]
            token_type_ids=None,
            labels=None,
            images=None,
            aux_imgs=None,
            rcnn_imgs=None,
    ):
        # 批次32
        bsz = input_ids.size(0)
        # zs IFA
        # (32,61)(32,61)(32.61)(32,3,224,224)(32,3,3,224,224)(32,3,3,224,224)
        output, loss,output_v = self.model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,

                                  pixel_values=images,
                                  aux_values=aux_imgs,
                                  rcnn_values=rcnn_imgs,
                                  return_dict=True, )

        # [32,80,768],[32,768]
        last_hidden_state, pooler_output = output.last_hidden_state, output.pooler_output
        bsz, seq_len, hidden_size = last_hidden_state.shape

        entity_hidden_state = torch.Tensor(bsz, 2*hidden_size) # batch, 2*hidden
        # entity_hidden_state = torch.Tensor(bsz, hidden_size)  # batch, 2*hidden

        # HSN
        hzk_hidden_matrix = torch.Tensor(hidden_size, hidden_size)
        hzk_hidden_matrix_t = torch.Tensor(hidden_size, hidden_size)

        for i in range(bsz):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            # [768]
            head_hidden = last_hidden_state[i, head_idx, :].squeeze()
            tail_hidden = last_hidden_state[i, tail_idx, :].squeeze()

            # hzk
            # hzk_hidden_matrix = torch.matmul(head_hidden.reshape(hidden_size, 1),
            #                                  tail_hidden.reshape(1, hidden_size))
            # x = (torch.mean(hzk_hidden_matrix, dim=1, keepdim=True))
            # y = (torch.mean(hzk_hidden_matrix, dim=0, keepdim=True))

            # hzk2
            # hzk_hidden_matrix=torch.matmul(head_hidden.reshape(hidden_size,1),head_hidden.reshape(1,hidden_size))
            # x_h = (torch.mean(hzk_hidden_matrix, dim=1, keepdim=True))
            # y_h = (torch.mean(hzk_hidden_matrix, dim=0, keepdim=True))
            # hzk_hidden_matrix_t=torch.matmul(tail_hidden.reshape(hidden_size,1),tail_hidden.reshape(1,hidden_size))
            # x_t = (torch.mean(hzk_hidden_matrix, dim=1, keepdim=True))
            # y_t = (torch.mean(hzk_hidden_matrix, dim=0, keepdim=True))

            # [1536]0
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)

            # hzk1（768，class，entity）
            # entity_hidden_state[i] = (torch.add(torch.matmul(hzk_hidden_matrix, x),
            #                                     torch.matmul(hzk_hidden_matrix, y.T)) / 2).view(-1)
            # print(entity_hidden_state)

        #     hzk2
        #     entity_hidden_state[i] = torch.cat([(torch.add(torch.matmul(hzk_hidden_matrix,x_h),torch.matmul(hzk_hidden_matrix,y_h.T))/2).view(-1), (torch.add(torch.matmul(hzk_hidden_matrix_t,x_t),torch.matmul(hzk_hidden_matrix_t,y_t.T))/2).view(-1),], dim=-1)

        # (32,1536)
        entity_hidden_state = entity_hidden_state.to(self.args.device)
        # (32,23)
        logits = self.classifier(entity_hidden_state)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            # loss
            # return loss + loss_fn(logits, labels.view(-1)), logits
            return loss_fn(logits, labels.view(-1)), logits
        return logits




        # # hzk
        # # (32,61)(32,61)(32.61)(32,3,224,224)(32,3,3,224,224)(32,3,3,224,224)
        # output,loss = self.model(input_ids=input_ids,
        #                     attention_mask=attention_mask,
        #                     token_type_ids=token_type_ids,
        #
        #                     pixel_values=images,
        #                     aux_values=aux_imgs,
        #                     rcnn_values=rcnn_imgs,
        #                     return_dict=True,)
        #
        #
        # # (32,23)
        # logits = output
        # if labels is not None:
        #     loss_fn = nn.CrossEntropyLoss()
        #     return loss+loss_fn(logits, labels.view(-1)), logits
        # return logits
class IFAREModel_cat(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(IFAREModel_cat, self).__init__()

        self.args = args

        # zs
        self.vision_config = CLIPConfig.from_pretrained(self.args.vit_name).vision_config
        self.text_config = BertConfig.from_pretrained(self.args.bert_name)

        clip_model_dict = CLIPModel.from_pretrained(self.args.vit_name).vision_model.state_dict()
        bert_model_dict = BertModel.from_pretrained(self.args.bert_name).state_dict()

        print(self.vision_config)
        print(self.text_config)

        # for re
        self.vision_config.device = args.device
        self.model = IFAModel(self.vision_config, self.text_config,self.args)


        # load:
        vision_names, text_names = [], []
        model_dict = self.model.state_dict()

        # zs
        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
            elif 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '')
                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = bert_model_dict[text_name]
        assert len(vision_names) == len(clip_model_dict) and len(text_names) == len(bert_model_dict), \
                    (len(vision_names), len(text_names), len(clip_model_dict), len(bert_model_dict))
        self.model.load_state_dict(model_dict)

        # zs
        self.model.resize_token_embeddings(len(tokenizer))
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer
        # self.classifier = nn.Linear(self.text_config.hidden_size*2, num_labels)
        self.classifier = nn.Linear(self.text_config.hidden_size*2, num_labels)

        self.dropout = nn.Dropout(0.5)
        # self.model = HZK(args)
        self.ScaledDotProductAttention= ScaledDotProductAttention(768)

    def forward(
            self,
            # [32,80]
            input_ids=None,
            # [32,80]
            attention_mask=None,
            # [32,80]
            token_type_ids=None,
            labels=None,
            images=None,
            aux_imgs=None,
            rcnn_imgs=None,
    ):
        #批次32
        bsz = input_ids.size(0)
        # zs IFA
        # (32,61)(32,61)(32.61)(32,3,224,224)(32,3,3,224,224)(32,3,3,224,224)
        output,loss,output_v= self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,

                            pixel_values=images,
                            aux_values=aux_imgs,
                            rcnn_values=rcnn_imgs,
                            return_dict=True,)


        # [32,80,768],[32,768]
        last_hidden_state= output.last_hidden_state
        pooler_output=output.pooler_output
        bsz, seq_len, hidden_size = last_hidden_state.shape
        # v
        last_hidden_state_v,pooler_output_v= output_v.last_hidden_state,output_v.pooler_output
        entity_hidden_state = torch.Tensor(bsz, hidden_size * 2)  # batch, 2*hidden
        last_hidden_state_new= torch.Tensor(bsz,61,hidden_size)
        for i in range(bsz):

            last_hidden_state_new[i,:,:],none=self.ScaledDotProductAttention(last_hidden_state[i,:,:],last_hidden_state_v.last_hidden_state[i,:,:],last_hidden_state_v.last_hidden_state[i,:,:])


            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            # [768]
            head_hidden = last_hidden_state_new[i, head_idx, :].squeeze()
            tail_hidden = last_hidden_state_new[i, tail_idx, :].squeeze()


            # [1536]0
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)



        # (32,1536)
        entity_hidden_state = entity_hidden_state.to(self.args.device)
        # (32,23)
        logits = self.classifier(entity_hidden_state)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            # loss
            return loss+loss_fn(logits, labels.view(-1)), logits
            # return loss_fn(logits, labels.view(-1)), logits
        return logits

class IFAREModel_recon(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(IFAREModel_recon, self).__init__()

        self.args = args

        # zs
        self.vision_config = CLIPConfig.from_pretrained(self.args.vit_name).vision_config
        self.text_config = BertConfig.from_pretrained(self.args.bert_name)

        clip_model_dict = CLIPModel.from_pretrained(self.args.vit_name).vision_model.state_dict()
        bert_model_dict = BertModel.from_pretrained(self.args.bert_name).state_dict()

        print(self.vision_config)
        print(self.text_config)

        # for re
        self.vision_config.device = args.device
        self.model = IFAModel(self.vision_config, self.text_config,self.args)


        # load:
        vision_names, text_names = [], []
        model_dict = self.model.state_dict()

        # zs
        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
            elif 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '')
                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = bert_model_dict[text_name]
        assert len(vision_names) == len(clip_model_dict) and len(text_names) == len(bert_model_dict), \
                    (len(vision_names), len(text_names), len(clip_model_dict), len(bert_model_dict))
        self.model.load_state_dict(model_dict)

        # zs
        self.model.resize_token_embeddings(len(tokenizer))
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer
        # self.classifier = nn.Linear(self.text_config.hidden_size*2, num_labels)
        self.classifier = nn.Linear(self.text_config.hidden_size*2, num_labels)

        self.dropout = nn.Dropout(0.5)
        # self.model = HZK(args)
        self.ScaledDotProductAttention= ScaledDotProductAttention(768)

    def forward(
            self,
            # [32,80]
            input_ids=None,
            # [32,80]
            attention_mask=None,
            # [32,80]
            token_type_ids=None,
            labels=None,
            images=None,
            aux_imgs=None,
            rcnn_imgs=None,
    ):
        #批次32
        bsz = input_ids.size(0)
        # zs IFA
        # (32,61)(32,61)(32.61)(32,3,224,224)(32,3,3,224,224)(32,3,3,224,224)
        output,a,b,c,output_v,T_recon,V_recon= self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,

                            pixel_values=images,
                            aux_values=aux_imgs,
                            rcnn_values=rcnn_imgs,
                            return_dict=True,)


        # [32,80,768],[32,768]
        last_hidden_state= T_recon
        pooler_output=output.pooler_output
        bsz, seq_len, hidden_size = last_hidden_state.shape
        # v
        # last_hidden_state_v,pooler_output_v= output_v.last_hidden_state,output_v.pooler_output
        last_hidden_state_v=V_recon


        entity_hidden_state = torch.Tensor(bsz, hidden_size * 2)  # batch, 2*hidden
        last_hidden_state_new= torch.Tensor(bsz,61,hidden_size)
        for i in range(bsz):

            last_hidden_state_new[i,:,:],none=self.ScaledDotProductAttention(last_hidden_state[i,:,:],last_hidden_state_v[i,:,:],last_hidden_state_v[i,:,:])


            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            # [768]
            head_hidden = last_hidden_state_new[i, head_idx, :].squeeze()
            tail_hidden = last_hidden_state_new[i, tail_idx, :].squeeze()


            # [1536]0
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)



        # (32,1536)
        entity_hidden_state = entity_hidden_state.to(self.args.device)
        # (32,23)
        logits = self.classifier(entity_hidden_state)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            # loss
            # return loss_fn(logits, labels.view(-1)), logits
            return a+b+c+loss_fn(logits, labels.view(-1)), logits,a,b,c
        return logits

class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 1)
        self.tanh = nn.Tanh()

    def forward(self, q, k, v):
        # 求g
        g_fc1 = self.fc1(v)
        g_tanh = self.tanh(g_fc1)
        g_fc2 = self.fc2(g_tanh)
        g = g_fc2.squeeze()
        # qk
        scores = torch.matmul(q, k.transpose(-1, -2))
        # 求s
        # 对注意力分数进行缩放处理
        scores_scaled = scores / torch.sqrt(torch.tensor(q.shape[1]).float())
        # 对注意力分数进行softmax归一化
        s = F.softmax(scores_scaled, dim=1)
        # weights
        attn_scores = torch.add(torch.mul(scores, (1 - g)), torch.mul(torch.mean(s, dim=1), g)) / self.scale
        attn_weights = self.softmax(attn_scores)
        attn_weights = self.dropout(attn_weights)
        # output
        attn_output = torch.matmul(attn_weights, v)

        return attn_output, attn_weights