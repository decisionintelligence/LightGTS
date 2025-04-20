
__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from .layers.pos_encoding import *
from .layers.basics import *
from .layers.attention import *
from src.models.layers.decoder_cnn import Decoder

import torch.fft as fft
import math
import numpy as np
from einops import reduce, rearrange, repeat
from src.callback.decompose import st_decompose

            
# Cell
class LightGTS(nn.Module):
    """
    Output dimension: 
         [bs x target_dim x nvars] for prediction
         [bs x target_dim] for regression
         [bs x target_dim] for classification
         [bs x num_patch x n_vars x patch_len] for pretrain
    """
    def __init__(self, c_in:int, target_dim:int, patch_len:int, stride:int, target_patch_len: int,num_patch:int, mask_mode:str = 'patch',mask_nums:int = 3,
                 e_layers:int=3, d_layers:int=3, d_model=128, n_heads=16, d_ff:int=256, img_size=64,
                 norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", 
                 res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='sincos', learn_pe:bool=False, head_dropout = 0, 
                 head_type = "prediction", y_range:Optional[tuple]=None):

        super().__init__()
        assert head_type in ['pretrain', 'prediction', 'regression', 'classification'], 'head type should be either pretrain, prediction, or regression'

        # Basic
        self.num_patch = num_patch
        self.target_dim=target_dim
        self.out_patch_num = math.ceil(target_dim / patch_len)
        self.target_patch_len = target_patch_len
        self.quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Embedding
        self.embedding = nn.Linear(self.target_patch_len, d_model, bias=False)
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, 1, d_model),requires_grad=True)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=e_layers, 
                                    store_attn=store_attn)
        
        # Decoder
        self.decoder = Decoder(d_layers, patch_len=patch_len, d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)

        # Head
        self.n_vars = c_in
        self.head_type = head_type
        self.mask_mode = mask_mode
        self.mask_nums = mask_nums
        self.d_model  = d_model

        if head_type == "pretrain":
            self.head = PretrainHead(d_model, self.target_patch_len, len(self.quantiles), head_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == "prediction":
            self.head = decoder_PredictHead(d_model, self.target_patch_len, head_dropout)


    def decoder_predict(self, bs, n_vars, dec_cross):
        """
        dec_cross: tensor [bs x  n_vars x num_patch x d_model]
        """
        # dec_in = self.decoder_embedding.expand(bs, n_vars, self.out_patch_num, -1)
        dec_in = dec_cross[:,:,-1,:].unsqueeze(2).expand(-1,-1,self.out_patch_num,-1)
        weights = self.get_dynamic_weights(self.out_patch_num).to(dec_in.device)
        dec_in = dec_in * weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # dec_in = dec_in + self.pos[-self.out_patch_num:,:]
        decoder_output = self.decoder(dec_in, dec_cross)
        decoder_output = decoder_output.transpose(2,3)

        return decoder_output
    
    def get_dynamic_weights(self, n_preds):
        """
        Generate dynamic weights for the replicated tokens. This example uses a linearly decreasing weight.
        You can modify this to use other schemes like exponential decay, sine/cosine, etc.
        """
        # Linearly decreasing weights from 1.0 to 0.5 (as an example)
        weights = torch.linspace(1.0, 0.5, n_preds)
        return weights


    def forward(self, z):                             
        """
        z_masked : tensor [bs x num_patch x n_vars x patch_len x mask_nums]
        z_orginal : tensor [bs x num_patch x n_vars x patch_len]
        """   

        z_masked,z_original = z
        bs, num_patch, n_vars, patch_len, mask_nums = z_masked.shape 
        z_masked = z_masked.permute(4,0,1,2,3).reshape(bs*mask_nums, num_patch, n_vars, patch_len)
        z = torch.cat((z_original, z_masked), dim=0) 
        z = resize(z, target_patch_len=self.target_patch_len)

        # tokenizer
        cls_tokens = self.cls_embedding.expand(bs*(mask_nums + 1), n_vars, -1, -1)
        z = self.embedding(z).permute(0,2,1,3) # [bs*(1+mask_nums) x n_vars x num_patch x d_model]
        z = torch.cat((cls_tokens, z), dim=2)  # [bs*(1+mask_nums) x n_vars x (1 + num_patch) x d_model]
        # z = self.drop_out(z + self.pos[:1 + self.num_patch, :])

        # encoder 
        z = torch.reshape(z, (-1, 1 + self.num_patch, self.d_model)) # [bs*(1+mask_nums)*n_vars x num_patch x d_model]
        z = self.encoder(z)
        z = torch.reshape(z, (-1, n_vars, 1 + self.num_patch, self.d_model)) # [bs*(1+mask_nums), n_vars x num_patch x d_model]
        z_masked = z[bs:,:,:,:] # [bs*mask_nums, n_vars x num_patch x d_model]
        z_original = z[0:bs,:,:,:] # [bs, n_vars x num_patch x d_model]

        # aggregation
        if self.mask_mode == "multi" or self.mask_mode == "freq_multi":
            z_masked = z_masked.reshape(self.mask_nums, -1, n_vars, num_patch + 1, self.d_model,).mean(0).transpose(2,3)     # z_masked: [bs x nvars x d_model x num_patch]    

        # decoder_prediction
        z_predict = self.decoder_predict(bs, n_vars, z_original)
        z_predict = self.head(z_predict)  # [bs x num_patch x nvars x patch_len x num_classes]
        z_predict = z_predict.permute(0,1,3,2,4) # [bs x num_patch x patch_len x n_vars]
        z_predict = z_predict.reshape(z_predict.shape[0],-1,z_predict.shape[3],z_predict.shape[4])
        z_predict = z_predict[:,:self.target_dim, :, :]  

        # recontruction
        z_reconstruct = self.head(z_masked[:,:,:,1:]) # [bs x num_patch x nvars x patch_len x num_classes]
        z_reconstruct = z_reconstruct[:,:,:,:, :].mean(4)

        # z: [bs x target_dim x nvars x num_classes] for prediction
        #    [bs x num_patch x n_vars x patch_len x num_classes] for pretrain

        return z_reconstruct, z_predict


class PretrainHead(nn.Module):
    def __init__(self, d_model, patch_len, num_classes, dropout):
        """
        d_model: Dimension of model hidden state.
        patch_len: Length of each patch.
        num_classes: Number of regression targets (or classes for classification).
        dropout: Dropout rate.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.patch_len = patch_len
        
        # Separate linear layers for each class
        self.linears = nn.ModuleList([nn.Linear(d_model, patch_len, bias=False) for _ in range(num_classes)])

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x num_classes x patch_len]
        """
        x = x.transpose(2, 3)  # [bs x nvars x num_patch x d_model]
        outputs = []

        # Apply each linear layer to the input
        for linear in self.linears:
            output = linear(self.dropout(x)).unsqueeze(-1)  # [bs x nvars x num_patch x patch_len x 1]
            outputs.append(output)

        x = torch.cat(outputs, dim=-1)  # [bs x nvars x num_patch x patch_len x num_classes]
        x = x.permute(0, 2, 1, 3, 4)  # [bs x num_patch x nvars x patch_len x num_classes]
        return x


class decoder_PredictHead(nn.Module):
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        x: tensor [bs x nvars x d_model x num_patch]
        output: tensor [bs x nvars x num_patch x patch_len]
        """

        x = x.transpose(2,3)                     # [bs x nvars x num_patch x d_model]
        x = self.linear( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
        x = x.permute(0,2,3,1)                  # [bs x num_patch x patch_len x nvars]
        return x.reshape(x.shape[0],-1,x.shape[3])


class PatchTSTEncoder(nn.Module):
    def __init__(self, c_in, num_patch, patch_len, n_embedding=32,
                 n_layers=3, d_model=128, n_heads=16, shared_embedding=True,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding
        self.n_embedding=n_embedding         

        # Input encoding: projection of feature vectors onto a d-dim vector space
        if not shared_embedding: 
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars): self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model)     


        #(
        #discrete encoding: projection of feature vectors onto a discrete vector space
        if self.n_embedding!=0:
            self.W_D = nn.Linear(patch_len, d_model)
            self.vq_embedding = nn.Embedding(self.n_embedding, d_model)
            self.vq_embedding.weight.data.uniform_(-1.0 / d_model,
                                                1.0 / d_model)

        # if self.n_embedding!=0:
        #     self.W_D_s = nn.Linear(patch_len, d_model)
        #     self.W_D_t = nn.Linear(patch_len, d_model)
        #     self.W_D_r = nn.Linear(patch_len, d_model)
        #     self.vq_embedding_s = nn.Embedding(self.n_embedding, d_model)
        #     self.vq_embedding_s.weight.data.uniform_(-1.0 / d_model,
        #                                         1.0 / d_model)
        #     self.vq_embedding_t = nn.Embedding(self.n_embedding, d_model)
        #     self.vq_embedding_t.weight.data.uniform_(-1.0 / d_model,
        #                                         1.0 / d_model)
            
        #)

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, 
                                    store_attn=store_attn)

    def forward(self, x) -> Tensor:       
        """
        x: tensor [bs x num_patch x nvars x patch_len]
        """
        bs, num_patch, n_vars, patch_len = x.shape      # x: [bs x num_patch x nvars x d_model]
        # Input encoding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars): 
                z = self.W_P[i](x[:,:,i,:])
                x_out.append(z)
            x1 = torch.stack(x_out, dim=2)
        else:
            x1 = self.W_P(x)                                                    
        x1 = x1.transpose(1,2)
        #(
        # if self.n_embedding!=0:
        #     decompose = st_decompose(kernel_size=3)

        #     #svq
        #     trend, season, res = decompose(x.transpose(2, 3).reshape([x.shape[0],x.shape[1]*x.shape[3],x.shape[2]]))
        #     trend=trend.reshape(x.shape[0],x.shape[1],x.shape[3],x.shape[2]).transpose(2, 3)
        #     season=season.reshape(x.shape[0],x.shape[1],x.shape[3],x.shape[2]).transpose(2, 3)
        #     res=res.reshape(x.shape[0],x.shape[1],x.shape[3],x.shape[2]).transpose(2, 3)

        #     #cvq
        #     trend, season, res = decompose(x.transpose(1, 2).reshape([x.shape[0],x.shape[2],x.shape[1]*x.shape[3]]))
        #     trend=trend.reshape(x.shape[0],x.shape[2],x.shape[1],x.shape[3]).transpose(1, 2)
        #     season=season.reshape(x.shape[0],x.shape[2],x.shape[1],x.shape[3]).transpose(1, 2)
        #     res=res.reshape(x.shape[0],x.shape[2],x.shape[1],x.shape[3]).transpose(1, 2)

        #     season=self.W_D_s(season).transpose(1, 3)
        #     trend=self.W_D_t(trend).transpose(1, 3)
        #     res=self.W_D_r(res).transpose(1, 3)
        #     x2=season+trend+res
        #     embedding_s = self.vq_embedding_s.weight.data
        #     embedding_t = self.vq_embedding_t.weight.data
        #     N, C, H, W = x2.shape
        #     K, _ = embedding_s.shape
        #     embedding_s_broadcast = embedding_s.reshape(1, K, C, 1, 1)
        #     embedding_t_broadcast = embedding_t.reshape(1, K, C, 1, 1)
        #     season_broadcast = season.reshape(N, 1, C, H, W)
        #     trend_broadcast = trend.reshape(N, 1, C, H, W)
        #     distance_s = torch.sum((embedding_s_broadcast - season_broadcast) ** 2, 2)
        #     distance_t = torch.sum((embedding_t_broadcast - trend_broadcast) ** 2, 2)
        #     nearest_neighbor_s = torch.argmin(distance_s, 1)
        #     nearest_neighbor_t = torch.argmin(distance_t, 1)
        #     xq_s = self.vq_embedding_s(nearest_neighbor_s).permute(0, 3, 1, 2)
        #     xq_t = self.vq_embedding_t(nearest_neighbor_t).permute(0, 3, 1, 2)
        #     xq=xq_s+xq_t+res

        if self.n_embedding!=0:
            x2 = self.W_D(x)
            x2 = x2.transpose(1, 3)
            embedding = self.vq_embedding.weight.data
            N, C, H, W = x2.shape
            K, _ = embedding.shape
            embedding_broadcast = embedding.reshape(1, K, C, 1, 1)
            x2_broadcast = x2.reshape(N, 1, C, H, W)
            distance = torch.sum((embedding_broadcast - x2_broadcast) ** 2, 2)
            
            nearest_neighbor = torch.argmin(distance, 1)
            xq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)

            
            # soft_vq=3
            # ds,nearest_neighbors=torch.topk(distance, soft_vq, dim=1, largest = False)
            # xq =self.vq_embedding(nearest_neighbors[:,0,:,:])*torch.unsqueeze((torch.sum(ds,dim=1)-ds[:,0,:,:]),3)
            # for i in range(1,soft_vq):
            #     xq += self.vq_embedding(nearest_neighbors[:,i,:,:])*torch.unsqueeze((torch.sum(ds,dim=1)-ds[:,i,:,:]),3)
            # xq=(xq/((soft_vq-1)*torch.unsqueeze(torch.sum(ds,dim=1),3))).permute(0, 3, 1, 2)

            # make C to the second dim
            x2 = x2.transpose(1, 3)
            x2 = x2.transpose(1, 2)
            xq = xq.transpose(1, 3)
            xq = xq.transpose(1, 2)
            # stop gradient
            decoder_input = x2 + (xq - x2).detach()

            u = torch.reshape(decoder_input+x1, (bs*n_vars, num_patch, self.d_model) )              # u: [bs * nvars x num_patch x d_model]
            u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x num_patch x d_model]

            # Encoder
            z = self.encoder(u)                                                      # z: [bs * nvars x num_patch x d_model]
            z = torch.reshape(z, (-1,n_vars, num_patch, self.d_model))               # z: [bs x nvars x num_patch x d_model]
            z = z.permute(0,1,3,2)                                                # z: [bs x nvars x d_model x num_patch]

            return z, x2, xq
        #)  
        else: 
                                                  # x: [bs x nvars x num_patch x d_model]        
            u = torch.reshape(x1, (bs*n_vars, num_patch, self.d_model) )              # u: [bs * nvars x num_patch x d_model]
            u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x num_patch x d_model]
            
            # Encoder
            z = self.encoder(u)                                                      # z: [bs * nvars x num_patch x d_model]
            z = torch.reshape(z, (-1,n_vars, num_patch, self.d_model))               # z: [bs x nvars x num_patch x d_model]
            z = z.permute(0,1,3,2)                                                 # z: [bs x nvars x d_model x num_patch]

            return z
    
    
# Cell
class TSTEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, 
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor):
        """
        src: tensor [bs x q_len x d_model]
        """
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores)
            return output
        else:
            for mod in self.layers: output = mod(output)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, 
                activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None):
        """
        src: tensor [bs x q_len x d_model]
        """
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev)
        else:
            src2, attn = self.self_attn(src, src, src)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src



def resize(x, target_patch_len):
    '''
    x: tensor [bs x num_patch x n_vars x patch_len]]
    '''
    bs, num_patch, n_vars, patch_len = x.shape
    x = x.reshape(bs*num_patch, n_vars, patch_len)
    x = F.interpolate(x, size=target_patch_len, mode='linear')
    return x.reshape(bs, num_patch, n_vars, target_patch_len)
