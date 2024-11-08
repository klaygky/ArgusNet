import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import RobertaModel, ViTModel, AutoTokenizer
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
from utils.libs import *
from attention import MultiHeadAttention, ScaledDotProductAttention


class MMOD(nn.Module):

    def __init__(self, parser):
        super(MMOD, self).__init__()
        self.parser = parser
        self.grad_clip = parser.grad_clip
        self.roberta_tokenizer = AutoTokenizer.from_pretrained('../roberta-base')
        # self.state_encoder = KAN([11, 128, 384]).to(parser.device)
        self.text_3d_encoder = RobertaModel.from_pretrained('../roberta-base', add_pooling_layer=False)
        self.img_encoder = ViTModel.from_pretrained('../vit-base-patch16-224', add_pooling_layer=False)
        self.att_text = MultiHeadAttention(768, 64, 64, 4, 0.3)
        self.att_image = MultiHeadAttention(768, 64, 64, 4, 0.3)
        self.att_cross = MultiHeadAttention(768, 64, 64, 4, 0.3)
        self.text_encoder = RobertaModel.from_pretrained('../roberta-base', add_pooling_layer=False)
        self.img_3d_fusion = Img_3d_Fusion(768, 768, 512)

        self.norm = nn.LayerNorm(768, eps=1e-6)  # Layer Normalization
        self.pipeline = CrossModalFusionModule(parser)
        self.classifier = nn.Sequential(
            nn.Linear(parser.d_model * 2, parser.d_model),
            nn.ReLU(),
            nn.Linear(parser.d_model, 1)
        )

        if torch.cuda.is_available():
            self.text_3d_encoder.cuda()        
            self.img_encoder.cuda()     
            self.text_encoder.cuda()     
            self.img_3d_fusion.cuda() 
            self.pipeline.cuda()
            self.classifier.cuda()           
            cudnn.deterministic = True     
            cudnn.benchmark = True         

        params = list(self.text_3d_encoder.parameters())      
        params += list(self.img_encoder.parameters())   
        params += list(self.text_encoder.parameters())
        params += list(self.img_3d_fusion.parameters())  
        params += list(self.pipeline.parameters())    
        params += list(self.classifier.parameters())   

        self.params = params  

        decay_factor = 2e-4 

        self.optimizer = torch.optim.AdamW([
            {'params': self.text_3d_encoder.parameters(), 'lr': parser.learning_rate * 0.1},
            {'params': self.img_encoder.parameters(), 'lr': parser.learning_rate * 0.1},
            {'params': self.text_encoder.parameters(), 'lr': parser.learning_rate * 0.1},
            {'params': self.img_3d_fusion.parameters(), 'lr': parser.learning_rate},
            {'params': self.pipeline.parameters(), 'lr': parser.learning_rate},
            {'params': self.classifier.parameters(), 'lr': parser.learning_rate},
        ],
            lr=parser.learning_rate, weight_decay=decay_factor)

        # self.optimizer = torch.optim.AdamW([
        #     {'params': self.img_3d_fusion.parameters(), 'lr': parser.learning_rate},
        #     {'params': self.pipeline.parameters(), 'lr': parser.learning_rate},
        #     {'params': self.classifier.parameters(), 'lr': parser.learning_rate},
        # ], lr=parser.learning_rate, weight_decay=decay_factor)



    def forward(self, images, text_3d, queries):
        image_features = self.img_encoder(images).last_hidden_state #目标的2D特征信息
        text_3d_inputs = self.roberta_tokenizer(text_3d, return_tensors='pt', padding=True, truncation=True).to(self.parser.device)
        text_3d_cls = self.text_3d_encoder(**text_3d_inputs).last_hidden_state[:, 0, :].unsqueeze(1) #目标的3D文本信息
        fused = self.img_3d_fusion(image_features, text_3d_cls).to(self.parser.device)  #融合2D和3D信息得到完整的目标信息

        queries_inputs = self.roberta_tokenizer(queries, return_tensors='pt', padding=True, truncation=True).to(self.parser.device)
        queries_features = self.text_encoder(**queries_inputs).last_hidden_state
        # att_queries = self.att_text(queries_features, queries_features, queries_features)
        queries_features = self.norm(queries_features)

        image_embeddings, text_embeddings = self.pipeline(fused, queries_features)
        classification_input = torch.cat([image_embeddings, text_embeddings], dim=1)  # (2*bs, d_model*2)
        text_embeddings = F.normalize(text_embeddings, dim=1)
        image_embeddings = F.normalize(image_embeddings, dim=1)
        logits = self.classifier(classification_input).squeeze(1)  # (2*bs,)
        return text_embeddings, image_embeddings, logits