import torch
import torch.nn as nn

from layers.AutoCorrelation import AutoCorrelationLayer, AutoCorrelation
from layers.Myformer_EncDec import series_decomp, mixture_series_decomp
from layers.Transformer_EncDec import Encoder, EncoderLayer, Decoder, DecoderLayer

from layers.Embed import MyDataEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len

        kernel_size = configs.multi_moving_avg
        if isinstance(kernel_size, list):
            self.decomposition = mixture_series_decomp(kernel_size)
        else:
            self.decomposition = series_decomp(kernel_size)

        self.trend_linear = nn.Linear(self.seq_len, self.pred_len);

        self.season_enc_embedding = MyDataEmbedding(configs.enc_in, configs.d_model, self.seq_len,configs.embed, configs.freq,
                                                configs.dropout)

        self.season_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.season_dec_embedding = MyDataEmbedding(configs.dec_in, configs.d_model, self.pred_len+self.label_len,configs.embed, configs.freq,
                                           configs.dropout)
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        zeros = torch.zeros([x_dec.shape[0], self.pred_len,
                             x_dec.shape[2]], device=x_enc.device)
        # Kernal-Mixture Decomposition
        seasonal, trend = self.decomposition(x_enc)

        seasonal_dec = torch.cat(
            [seasonal[:, -self.label_len:, :], zeros], dim=1)
        enc_out = self.season_enc_embedding(seasonal, x_mark_enc)
        enc_out, attns = self.season_encoder(enc_out, attn_mask=None)

        dec_out = self.season_dec_embedding(seasonal_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        seasonal_result = dec_out[:, -self.pred_len:, :]

        trend = trend.permute(0, 2, 1)
        trend_result = self.trend_linear(trend).permute(0,2,1)
        result = seasonal_result + trend_result
        return result

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # return dec_out[:, -self.pred_len:, :]  # [B, L, D]
            return dec_out
        # if self.task_name == 'imputation':
        #     dec_out = self.imputation(
        #         x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        #     return dec_out  # [B, L, D]
        # if self.task_name == 'anomaly_detection':
        #     dec_out = self.anomaly_detection(x_enc)
        #     return dec_out  # [B, L, D]
        # if self.task_name == 'classification':
        #     dec_out = self.classification(x_enc, x_mark_enc)
        #     return dec_out  # [B, N]
        return None
