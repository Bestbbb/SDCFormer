import torch
from torch import nn
from layers.Myformer_EncDec import mixture_series_decomp, series_decomp
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer, SeasonDistanceAttention
from layers.Embed import MyDataEmbedding, PatchEmbedding, MyPatchEmbedding


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class Model(nn.Module):
    def __init__(self, configs,patch_len=8,stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        padding = stride
        kernel_size = configs.multi_moving_avg
        if isinstance(kernel_size, list):
            self.decomposition = mixture_series_decomp(kernel_size)
        else:
            self.decomposition = series_decomp(kernel_size)

        self.patch_embedding = MyPatchEmbedding(
            configs.d_model, patch_len, stride, padding,configs.dropout)
        
        self.trend_linear = nn.Linear(self.seq_len, self.pred_len)
        self.trend_linear.weight = nn.Parameter(
                (1 / self.pred_len) * torch.ones([self.pred_len, self.seq_len]),
                requires_grad=True)

        self.season_enc_embedding = MyDataEmbedding(configs.enc_in, configs.d_model, self.seq_len,configs.embed, configs.freq,
                                                configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        SeasonDistanceAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # self.season_encoder = Encoder(
        #     [
        #         EncoderLayer(
        #             AttentionLayer(
        #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=configs.output_attention), configs.d_model, configs.n_heads),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation
        #         ) for l in range(configs.e_layers)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(configs.d_model)
        # )

        # self.season_dec_embedding = MyDataEmbedding(configs.dec_in, configs.d_model, self.pred_len+self.label_len,configs.embed, configs.freq,
        #                                    configs.dropout)
        # self.decoder = Decoder(
        #     [
        #         DecoderLayer(
        #             AttentionLayer(
        #                 FullAttention(True, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=False),
        #                 configs.d_model, configs.n_heads),
        #             AttentionLayer(
        #                 FullAttention(False, configs.factor, attention_dropout=configs.dropout,
        #                               output_attention=False),
        #                 configs.d_model, configs.n_heads),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation,
        #         )
        #         for l in range(configs.d_layers)
        #     ],
        #     norm_layer=torch.nn.LayerNorm(configs.d_model),
        #     projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        # )
        self.head_nf = configs.d_model * \
                    int((configs.seq_len - patch_len) / stride + 1) # 如果无padding就是+1,有padding就是+2
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc/stdev
        # seasonal, trend = self.decomposition(x_enc)


        # trend_means = trend.mean(1, keepdim=True).detach()
        # trend_std = torch.sqrt(
        #     torch.var(trend, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # trend = trend - trend_means
        # trend = trend / trend_std
        # seasonal = seasonal.clone()

        # do patching and embedding
        # seasonal = seasonal.permute(0, 2, 1)
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        # dec_out = dec_out * \
        #           (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # dec_out = dec_out + \
        #           (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # trend = trend.permute(0, 2, 1)
        # trend = trend.permute(0, 2, 1)
        # trend_result = self.trend_linear(trend).permute(0,2,1)
        # trend_result = trend_result * \
        #           (trend_std[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # trend_result = trend_result + \
        #           (trend_means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        result = dec_out


        result = result * \
                (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        result = result + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return result
        # zeros = torch.zeros([x_dec.shape[0], self.pred_len,
        #                      x_dec.shape[2]], device=x_enc.device)
        # seasonal, trend = self.decomposition(x_enc)
        # means = seasonal.mean(1, keepdim=True).detach()
        # seasonal = seasonal - means
        # stdev = torch.sqrt(
        #     torch.var(seasonal, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # seasonal /= stdev
        # seasonal = seasonal.permute(0, 2, 1)

        # enc_out, n_vars = self.patch_embedding(seasonal)
        # enc_out, attns = self.encoder(enc_out)
        # enc_out = torch.reshape(
        #     enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # # z: [bs x nvars x d_model x patch_num]
        # enc_out = enc_out.permute(0, 1, 3, 2)

        # # Decoder
        # dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        # dec_out = dec_out.permute(0, 2, 1)

        # dec_out = dec_out * \
        #           (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # dec_out = dec_out + \
        #           (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        # trend = trend.permute(0, 2, 1)
        # trend_result = self.trend_linear(trend).permute(0,2,1)
        # result = dec_out + trend_result
        # return result

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
