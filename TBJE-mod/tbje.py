from torch import nn
import torch.nn.functional as F
import torch

class TBJEBlock(nn.Module):

    def __init__(self,
                 internal_dim: int,
                 mlp_hidden_dim: int,
                 n_heads: int):
        super().__init__()

        ################# modules for cross attention for fusion ###############
        self.vid_ln1 = nn.LayerNorm(internal_dim, bias=False)
        self.txt_ln1 = nn.LayerNorm(internal_dim, bias=False)
        self.aud_ln1 = nn.LayerNorm(internal_dim, bias=False)

        self.vid_cross_attn = nn.MultiheadAttention(internal_dim, n_heads, batch_first=True, bias=False, dropout=.2)
        self.txt_cross_attn = nn.MultiheadAttention(internal_dim, n_heads, batch_first=True, bias=False, dropout=.2)
        self.aud_cross_attn = nn.MultiheadAttention(internal_dim, n_heads, batch_first=True, bias=False, dropout=.2)
        ########################################################################

        ################################# MLP ##################################
        self.vid_ln2 = nn.LayerNorm(internal_dim, bias=False)
        self.txt_ln2 = nn.LayerNorm(internal_dim, bias=False)
        self.aud_ln2 = nn.LayerNorm(internal_dim, bias=False)

        self.vid_mlp_in = nn.Linear(internal_dim, mlp_hidden_dim, bias=False)
        self.vid_mlp_out = nn.Linear(mlp_hidden_dim, internal_dim, bias=False)
        self.txt_mlp_in = nn.Linear(internal_dim, mlp_hidden_dim, bias=False)
        self.txt_mlp_out = nn.Linear(mlp_hidden_dim, internal_dim, bias=False)
        self.aud_mlp_in = nn.Linear(internal_dim, mlp_hidden_dim, bias=False)
        self.aud_mlp_out = nn.Linear(mlp_hidden_dim, internal_dim, bias=False)
        ########################################################################

        ########################### Self Attention #############################
        self.vid_ln3 = nn.LayerNorm(internal_dim, bias=False)
        self.txt_ln3 = nn.LayerNorm(internal_dim, bias=False)
        self.aud_ln3 = nn.LayerNorm(internal_dim, bias=False)

        self.vid_slf_attn = nn.MultiheadAttention(internal_dim, n_heads, batch_first=True, bias=False, dropout=.2)
        self.txt_slf_attn = nn.MultiheadAttention(internal_dim, n_heads, batch_first=True, bias=False, dropout=.2)
        self.aud_slf_attn = nn.MultiheadAttention(internal_dim, n_heads, batch_first=True, bias=False, dropout=.2)
        ########################################################################

    def forward(self, data_tuple):
        vid_data, txt_data, aud_data = data_tuple
        # cross attention stage
        vid_ln_out = self.vid_ln1(vid_data)
        txt_ln_out = self.txt_ln1(txt_data)
        aud_ln_out = self.aud_ln1(aud_data)
        vid_attn_out = self.vid_cross_attn(vid_ln_out, txt_ln_out, txt_ln_out, is_causal=False)
        txt_attn_out = self.txt_cross_attn(txt_ln_out, vid_ln_out + aud_ln_out, vid_ln_out + aud_ln_out, is_causal=False)
        aud_attn_out = self.aud_cross_attn(aud_ln_out, txt_ln_out, txt_ln_out, is_causal=False)
        # residual connection
        vid_data = vid_data + vid_attn_out[0]
        txt_data = txt_data + txt_attn_out[0]
        aud_data = aud_data + aud_attn_out[0]

        # MLP stage
        vid_ln_out = self.vid_ln2(vid_data).contiguous()
        txt_ln_out = self.txt_ln2(txt_data).contiguous()
        aud_ln_out = self.aud_ln2(aud_data).contiguous()
        vid_mlp_out = self.vid_mlp_out(F.relu(self.vid_mlp_in(vid_ln_out)))
        txt_mlp_out = self.txt_mlp_out(F.relu(self.txt_mlp_in(txt_ln_out)))
        aud_mlp_out = self.aud_mlp_out(F.relu(self.aud_mlp_in(aud_ln_out)))
        # residual connection
        vid_data = vid_data + vid_mlp_out
        txt_data = txt_data + txt_mlp_out
        aud_data = aud_data + aud_mlp_out

        # self attention stage
        vid_ln_out = self.vid_ln3(vid_data)
        txt_ln_out = self.txt_ln3(txt_data)
        aud_ln_out = self.aud_ln3(aud_data)
        vid_attn_out = self.vid_slf_attn(vid_ln_out, vid_ln_out, vid_ln_out, is_causal=False)
        txt_attn_out = self.txt_slf_attn(txt_ln_out, txt_ln_out, txt_ln_out, is_causal=False)
        aud_attn_out = self.aud_slf_attn(aud_ln_out, aud_ln_out, aud_ln_out, is_causal=False)
        # residual connection
        vid_data = vid_data + vid_attn_out[0]
        txt_data = txt_data + txt_attn_out[0]
        aud_data = aud_data + aud_attn_out[0]

        return vid_data, txt_data, aud_data


class TBJENew(nn.Module):

    def __init__(self,
                 video_dim: int,
                 text_dim: int,
                 audio_dim: int,
                 internal_dim: int,
                 n_heads: int,
                 mlp_hidden_dim: int,
                 n_layers: int):
        super().__init__()

        self.vid_initial_proj = nn.Linear(video_dim, internal_dim, bias=False)
        self.txt_initial_proj = nn.Linear(text_dim, internal_dim, bias=False)
        self.aud_initial_proj = nn.Linear(audio_dim, internal_dim, bias=False)

        self.vid_proj_ln = nn.LayerNorm(internal_dim, bias=False)
        self.txt_proj_ln = nn.LayerNorm(internal_dim, bias=False)
        self.aud_proj_ln = nn.LayerNorm(internal_dim, bias=False)

        self.blocks = [TBJEBlock(internal_dim, mlp_hidden_dim, n_heads) for _ in range(n_layers)]
        self.blocks = nn.Sequential(*self.blocks)

        self.final_proj1 = nn.Linear(3 * internal_dim, 128, bias=False)
        self.final_proj2 = nn.Linear(128, 1, bias=False)
        self.output_activation = nn.Hardtanh(-3.0, 3.0)

    def forward(self, vid_data, txt_data, aud_data):
        assert not torch.isnan(vid_data).any(), "NaN found in video data"
        assert not torch.isnan(txt_data).any(), "NaN found in text data"
        assert not torch.isnan(aud_data).any(), "NaN found in audio data"
        # assuming input is batch_size * seq_len * embed_dim
        vid_data = self.vid_proj_ln(self.vid_initial_proj(vid_data.contiguous()))
        txt_data = self.txt_proj_ln(self.txt_initial_proj(txt_data.contiguous()))
        aud_data = self.aud_proj_ln(self.aud_initial_proj(aud_data.contiguous()))

        vid_data, txt_data, aud_data = self.blocks((vid_data, txt_data, aud_data))
        vid_data = vid_data[:, -1, :]
        txt_data = txt_data[:, -1, :]
        aud_data = aud_data[:, -1, :]

        concat = torch.concat((vid_data, txt_data, aud_data), dim=1)
        res = self.final_proj2(F.relu(self.final_proj1(concat.contiguous())).contiguous())
        res = self.output_activation(res)
        return res
