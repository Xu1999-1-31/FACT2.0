import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Delay_encoder(nn.Module):
    def __init__(self):
        super(Delay_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),  # [16, 8, 8] (I-K+2P)/S  + 1  I = 8, K = 3, P = 1, S = 1
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # [16, 4, 4]
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),  # [8, 4, 4]
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=1, stride=1, padding=0),   # [8, 4, 4]
            nn.ReLU()
        )
        self.fc = nn.Linear(128, 64)  # 8x4x4 to 128x1
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x

class Delay_decoder(nn.Module):
    def __init__(self):
        super(Delay_decoder, self).__init__()
        self.fc = nn.Linear(64, 128)  # Adjusted size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=1, stride=1),  # [8, 4, 4]  (input - 1)*stride + output_padding – 2*padding + kernel_size
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # [16, 8, 8] 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, output_padding=0)  # [3, 8, 8]  
            #nn.Sigmoid()  # [3, 8, 8] value in [0, 1]
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 8, 4, 4)  # Reshape to [batch_size, 8, 4, 4]
        x = self.decoder(x)
        return x

class Delay_Autoencoder(nn.Module):
    def __init__(self):
        super(Delay_Autoencoder, self).__init__()
        self.encoder = Delay_encoder()
        self.decoder = Delay_decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Trans_encoder(nn.Module):
    def __init__(self):
        super(Trans_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # [64, 8, 8] (I-K+2P)/S  + 1  I = 8, K = 3, P = 1, S = 1
            nn.ReLU(),
            #nn.MaxPool2d(2, stride=2),  # [64, 4, 4]
            nn.Conv2d(64, 32, kernel_size=4, stride=2, padding=1),  # [32, 4, 4]
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),   # [16, 4, 4]
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=0),  # [8, 4, 4]
            nn.ReLU()
        )
        self.fc = nn.Linear(128, 64)  # 8x1x1 to 8x1
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x

class Trans_decoder(nn.Module):
    def __init__(self):
        super(Trans_decoder, self).__init__()
        self.fc = nn.Linear(64, 128)  # Adjusted size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=1, stride=1),  # [8, 4, 4]  (input - 1)*stride + output_padding – 2*padding + kernel_size
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # [16, 8, 8] 
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, output_padding=0)  # [3, 8, 8]  
            #nn.Sigmoid()  # [3, 8, 8] value in [0, 1]
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 8, 4, 4)  # Reshape to [batch_size, 8, 4, 4]
        x = self.decoder(x)
        return x


class Trans_Autoencoder(nn.Module):
    def __init__(self):
        super(Trans_Autoencoder, self).__init__()
        self.encoder = Trans_encoder()
        self.decoder = Trans_decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Cons_encoder(nn.Module):
    def __init__(self):
        super(Cons_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.ConvTranspose2d(3, 8, kernel_size=2, stride=2, padding=1), # [16, 4, 4] (input - 1)*stride + output_padding – 2*padding + kernel_size  2  
            nn.ReLU(),
            nn.ConvTranspose2d(8, 6, kernel_size=2, stride=2, padding=1),  # [8, 6, 6]
            nn.ReLU(),
            nn.ConvTranspose2d(6, 4, kernel_size=3, stride=1, padding=0),  # [4, 8, 8]
            nn.ReLU(),
        )
        self.fc = nn.Linear(256, 128)  # 256x1x1 to 256x1

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x

class Cons_decoder(nn.Module):
    def __init__(self):
        super(Cons_decoder, self).__init__()
        self.fc = nn.Linear(128, 256)  # Adjust size to match the intermediate dimensions
        self.decoder = nn.Sequential(
            nn.Conv2d(4, 6, kernel_size=5, stride=1, padding=1),  # [8, 6, 6] (I-K+2P)/S + 1   1 + 2 + 1
            nn.ReLU(),
            nn.Conv2d(6, 8, kernel_size=5, stride=1, padding=1), # [16, 4, 4]
            nn.ReLU(),
            nn.Conv2d(8, 3, kernel_size=2, stride=1, padding=0) # [3, 3, 3]
            #nn.Sigmoid()  # Output values in [0, 1]
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 4, 8, 8)  # Reshape to [batch_size, 64, 4, 4]
        x = self.decoder(x)
        return x

class Cons_Autoencoder(nn.Module):
    def __init__(self):
        super(Cons_Autoencoder, self).__init__()
        self.encoder = Cons_encoder()
        self.decoder = Cons_decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class MLP(torch.nn.Module):
    def __init__(self, *sizes, batchnorm=False, dropout=False):
        super().__init__()
        fcs = []
        for i in range(1, len(sizes)):
            fcs.append(torch.nn.Linear(sizes[i - 1], sizes[i]))
            if i < len(sizes) - 1:
                fcs.append(torch.nn.LeakyReLU(negative_slope=0.2))
                # fcs.append(torch.nn.ReLU())
                if dropout: fcs.append(torch.nn.Dropout(p=0.2))
                if batchnorm: fcs.append(torch.nn.BatchNorm1d(sizes[i])) # normalize every column
        self.layers = torch.nn.Sequential(*fcs)

    def forward(self, x):
        return self.layers(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class MMMC_Transformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_encoder_layers=3, num_decoder_layers=3, output_dim=2):
        super(MMMC_Transformer, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers,
                                          batch_first = True)
        self.mlp = MLP(d_model + 68, 256, 128, 64, 32, 16, 8, 4, output_dim)

    def forward(self, src, tgt, padding_mask, fpath, base_itf, pre_itf):
        # position encoding
        src_encoded = self.pos_encoder(src)
        tgt_encoded = self.pos_encoder(tgt)

        # src_encoded = src
        # tgt_encoded = tgt

        # transformer output
        transformer_output = self.transformer(src=src_encoded, tgt=tgt_encoded,
                                              src_key_padding_mask=padding_mask,
                                              tgt_key_padding_mask=padding_mask)

        transformer_output = transformer_output.transpose(1, 2)
        pooled_output = torch.nn.functional.max_pool1d(transformer_output, kernel_size=transformer_output.size(-1)).squeeze(-1)
        # print(pooled_output.size(), base_itf.size(), pre_itf.size(), fpath.size())
        
        mlp_input = torch.cat((pooled_output, base_itf, pre_itf, fpath), dim = 1)
        # MLP
        #aggregated_output = transformer_output.mean(dim=1)
        #output = self.mlp(aggregated_output)

        output = self.mlp(mlp_input)

        return output

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(ResidualMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.align_layers = nn.ModuleList()
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.align_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        for layer, align_layer in zip(self.layers, self.align_layers):
            residual = align_layer(x)
            x = F.relu(layer(x))
            x = x + residual
        return self.output_layer(x)


# training takes more time
class Res_MMMC_Transformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_encoder_layers=3, num_decoder_layers=3, output_dim=2):
        super(Res_MMMC_Transformer, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers,
                                          batch_first = True)
        self.res_mlp = ResidualMLP(d_model + 68, [256, 128, 64, 32, 16, 8, 4], output_dim)

    def forward(self, src, tgt, padding_mask, fpath, base_itf, pre_itf):
        # position encoding
        src_encoded = self.pos_encoder(src)
        tgt_encoded = self.pos_encoder(tgt)

        # src_encoded = src
        # tgt_encoded = tgt

        # transformer output
        transformer_output = self.transformer(src=src_encoded, tgt=tgt_encoded,
                                              src_key_padding_mask=padding_mask,
                                              tgt_key_padding_mask=padding_mask)

        transformer_output = transformer_output.transpose(1, 2)
        pooled_output = torch.nn.functional.max_pool1d(transformer_output, kernel_size=transformer_output.size(-1)).squeeze(-1)
        # print(pooled_output.size(), base_itf.size(), pre_itf.size(), fpath.size())
        
        mlp_input = torch.cat((pooled_output, base_itf, pre_itf, fpath), dim = 1)
        # MLP
        #aggregated_output = transformer_output.mean(dim=1)
        #output = self.mlp(aggregated_output)

        output = self.res_mlp(mlp_input)

        return output

class BidirectionalDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(BidirectionalDecoder, self).__init__()
        self.forward_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.backward_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

    def forward(self, tgt, memory, tgt_key_padding_mask):
        output_forward = self.forward_decoder(tgt, memory, tgt_key_padding_mask=tgt_key_padding_mask)

        tgt_reversed = torch.flip(tgt, [1])  # flip on the first dimension
        padding_reversed = torch.flip(tgt_key_padding_mask, [1])
        output_backward = self.backward_decoder(tgt_reversed, memory,  tgt_key_padding_mask=padding_reversed)
        # output_backward = torch.flip(output_backward, [1])  # flip back

        output = (output_forward + output_backward) / 2  

        return output

class MMMC_BidTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_encoder_layers=3, num_decoder_layers=3, output_dim=2, dim_feedforward=2048, dropout=0.1):
        super(MMMC_BidTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first = True)
        self.decoder = BidirectionalDecoder(decoder_layer, num_decoder_layers)
        self.mlp = MLP(d_model + 68, 256, 128, 64, 32, 16, 8, 4, output_dim)

    def forward(self, src, tgt, padding_mask, fpath, base_itf, pre_itf):
        src_encoded = self.pos_encoder(src)
        tgt_encoded = self.pos_encoder(tgt)
        memory = self.encoder(src_encoded, src_key_padding_mask=padding_mask)
        trans_output = self.decoder(tgt_encoded, memory, padding_mask)
        trans_output = trans_output.transpose(1, 2)
        pooled_output = torch.nn.functional.max_pool1d(trans_output, kernel_size=trans_output.size(-1)).squeeze(-1)
        
        mlp_input = torch.cat((pooled_output, base_itf, pre_itf, fpath), dim = 1)
        output = self.mlp(mlp_input)
        return output

class Res_MMMC_BidTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_encoder_layers=3, num_decoder_layers=3, output_dim=2, dim_feedforward=2048, dropout=0.1):
        super(Res_MMMC_BidTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout, batch_first = True)
        self.decoder = BidirectionalDecoder(decoder_layer, num_decoder_layers)
        self.mlp = MLP(d_model + 68, 256, 128, 64, 32, 16, 8, 4, output_dim)

    def forward(self, src, tgt, padding_mask, fpath, base_itf, pre_itf):
        src_encoded = self.pos_encoder(src)
        tgt_encoded = self.pos_encoder(tgt)
        memory = self.encoder(src_encoded, src_key_padding_mask=padding_mask)
        trans_output = self.decoder(tgt_encoded, memory, padding_mask)
        trans_output = trans_output.transpose(1, 2)
        pooled_output = torch.nn.functional.max_pool1d(trans_output, kernel_size=trans_output.size(-1)).squeeze(-1)
        
        base_slack_delay = fpath[:, :2]
        mlp_input = torch.cat((pooled_output, base_itf, pre_itf, fpath), dim = 1)
        output = self.mlp(mlp_input)
        output += base_slack_delay
        return output

class CrossHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(CrossHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm(x)
        return x

class Cross_Head_MMMC_Transformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_encoder_layers=3, num_decoder_layers=3, output_dim=2, dropout=0.1):
        super(Cross_Head_MMMC_Transformer, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers,
                                          batch_first=True, 
                                          dropout=dropout)
        self.cross_head_attn = CrossHeadAttention(d_model, num_heads=nhead, dropout=dropout)
        self.mlp = MLP(d_model + 68, 256, 128, 64, 32, 16, 8, 4, output_dim)

    def forward(self, src, tgt, padding_mask, fpath, base_itf, pre_itf):
        src_encoded = self.pos_encoder(src)
        tgt_encoded = self.pos_encoder(tgt)

        transformer_output = self.transformer(src=src_encoded, tgt=tgt_encoded,
                                              src_key_padding_mask=padding_mask,
                                              tgt_key_padding_mask=padding_mask)

        transformer_output = self.cross_head_attn(transformer_output)

        transformer_output = transformer_output.transpose(1, 2)
        pooled_output = torch.nn.functional.max_pool1d(transformer_output, kernel_size=transformer_output.size(-1)).squeeze(-1)

        mlp_input = torch.cat((pooled_output, base_itf, pre_itf, fpath), dim=1)
        output = self.mlp(mlp_input)

        return output

# Dynamic Attention
class DynamicAttention(nn.Module):
    def __init__(self, d_model):
        super(DynamicAttention, self).__init__()
        self.attn_weights = nn.Linear(d_model, 1, bias=False)

    def forward(self, x):
        weights = torch.softmax(self.attn_weights(x), dim=1)
        weighted_output = x * weights
        return weighted_output.sum(dim=1)

# Bidirectional Dynamic Decoder
class BidirectionalDynamicDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, d_model):
        super(BidirectionalDynamicDecoder, self).__init__()
        self.forward_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.backward_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.dynamic_attn = DynamicAttention(d_model)

    def forward(self, tgt, memory, tgt_key_padding_mask):
        forward_output = self.forward_decoder(tgt, memory, tgt_key_padding_mask=tgt_key_padding_mask)

        backward_tgt = torch.flip(tgt, [1])
        backward_output = self.backward_decoder(backward_tgt, memory, tgt_key_padding_mask=torch.flip(tgt_key_padding_mask, [1]))

        combined_output = (forward_output + backward_output) / 2
        dynamic_output = self.dynamic_attn(combined_output)
        return dynamic_output

# Full Transformer Model with Dynamic Attention and Bidirectional Decoder
class MmmcTransformerDynamicBidirectional(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_encoder_layers=3, num_decoder_layers=3, output_dim=2, dim_feedforward=2048, dropout=0.1):
        super(MmmcTransformerDynamicBidirectional, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Bidirectional Decoder with Dynamic Attention
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.decoder = BidirectionalDynamicDecoder(decoder_layer, num_decoder_layers, d_model)

        # Fully Connected Layer (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(d_model + 68, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, src, tgt, padding_mask, fpath, base_itf, pre_itf):
        # Positional Encoding
        src_encoded = self.pos_encoder(src)
        tgt_encoded = self.pos_encoder(tgt)

        # Encoder output
        memory = self.encoder(src_encoded, src_key_padding_mask=padding_mask)

        # Bidirectional Decoder with Dynamic Attention
        trans_output = self.decoder(tgt_encoded, memory, padding_mask)

        # Concatenate pooled output and additional features
        mlp_input = torch.cat((trans_output, base_itf, pre_itf, fpath), dim=1)

        # Fully Connected Layer
        output = self.mlp(mlp_input)
        return output
