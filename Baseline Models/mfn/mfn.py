"""
Code adapted from https://github.com/pliang279/MFN/blob/master/test_mosi.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mosei_dataset import MOSEIDataset
from torch.utils.data import DataLoader

def nan_check_hook(module, input, output):
    # Check all inputs for NaN values
    inputs = input if isinstance(input, (list, tuple)) else (input,)
    for inp in inputs:
        # If the input is a tensor, check it for NaN values
        if isinstance(inp, torch.Tensor) and torch.isnan(inp).any():
            raise ValueError(f"NaN detected in input of {module.__class__.__name__}")
    
    # Check all outputs for NaN values
    # Handle if output is a single tensor or a tuple/list of tensors
    outputs = output if isinstance(output, (list, tuple)) else (output,)
    for out in outputs:
        if isinstance(out, torch.Tensor) and torch.isnan(out).any():
            raise ValueError(f"NaN detected in output of {module.__class__.__name__}")
    
    return output

def init_lstm_weights(lstm):
    for name, param in lstm.named_parameters():
        if 'weight_ih' in name:
            # Xavier initialization for input-hidden weights
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            # Orthogonal initialization for hidden-hidden weights
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            # Initialize all biases to zero
            param.data.fill_(0)

CONFIG = {
    "input_dims": [711, 300, 74],
    "h_dims": [64, 64, 64],
    "memsize": 64,
    "windowsize": 2,
}
NN_CONFIG = {
    "shapes": 64,
    "drop": 0.5
}
GAMMA_CONFIG = {
    "shapes": 64,
    "drop": 0.5
}
OUT_CONFIG = {
    "shapes": 64,
    "drop": 0.5
}

class MFN3(nn.Module):
    def __init__(self,
                 config=CONFIG,
                 NN1Config=NN_CONFIG,
                 NN2Config=NN_CONFIG,
                 gamma1Config=GAMMA_CONFIG,
                 gamma2Config=GAMMA_CONFIG,
                 outConfig=OUT_CONFIG,
                 max_seqlen=None):
        super(MFN3, self).__init__()
        self.max_seqlen = max_seqlen
        [self.d_l,self.d_a,self.d_v] = config["input_dims"]
        [self.dh_l,self.dh_a,self.dh_v] = config["h_dims"]
        total_h_dim = self.dh_l+self.dh_a+self.dh_v
        self.mem_dim = config["memsize"]
        window_dim = config["windowsize"]
        output_dim = 1
        attInShape = total_h_dim*window_dim
        gammaInShape = attInShape+self.mem_dim
        final_out = total_h_dim+self.mem_dim
        h_att1 = NN1Config["shapes"]
        h_att2 = NN2Config["shapes"]
        h_gamma1 = gamma1Config["shapes"]
        h_gamma2 = gamma2Config["shapes"]
        h_out = outConfig["shapes"]
        att1_dropout = NN1Config["drop"]
        att2_dropout = NN2Config["drop"]
        gamma1_dropout = gamma1Config["drop"]
        gamma2_dropout = gamma2Config["drop"]
        out_dropout = outConfig["drop"]

        # self.ln_in = nn.LayerNorm((self.d_l + self.d_a + self.d_v,))

        self.lstm_l = nn.LSTMCell(self.d_l, self.dh_l, bias=False)
        self.lstm_a = nn.LSTMCell(self.d_a, self.dh_a, bias=False)
        self.lstm_v = nn.LSTMCell(self.d_v, self.dh_v, bias=False)
        init_lstm_weights(self.lstm_l)
        init_lstm_weights(self.lstm_a)
        init_lstm_weights(self.lstm_v)

        self.att1_fc1 = nn.Linear(attInShape, h_att1)
        self.att1_fc2 = nn.Linear(h_att1, attInShape)
        self.att1_dropout = nn.Dropout(att1_dropout)

        self.att2_fc1 = nn.Linear(attInShape, h_att2)
        self.att2_fc2 = nn.Linear(h_att2, self.mem_dim)
        self.att2_dropout = nn.Dropout(att2_dropout)

        self.gamma1_fc1 = nn.Linear(gammaInShape, h_gamma1)
        self.gamma1_fc2 = nn.Linear(h_gamma1, self.mem_dim)
        self.gamma1_dropout = nn.Dropout(gamma1_dropout)

        self.gamma2_fc1 = nn.Linear(gammaInShape, h_gamma2)
        self.gamma2_fc2 = nn.Linear(h_gamma2, self.mem_dim)
        self.gamma2_dropout = nn.Dropout(gamma2_dropout)

        self.out_fc1 = nn.Linear(final_out, h_out, bias=False)
        self.out_fc2 = nn.Linear(h_out, output_dim, bias=False)
        self.out_dropout = nn.Dropout(out_dropout)

        if torch.is_anomaly_enabled():
            print("Forward NaN check enabled for MFN")
            for module in self.modules():
                module.register_forward_hook(nan_check_hook)
        
    def forward(self,x):
        # assume x.shape == n x t x d, make it t x n x d
        x = x.transpose(0, 1)
        t, n, d = x.shape
        # x = x.reshape((t * n, d))
        # x = self.ln_in(x).reshape((t, n, d))

        x_l = x[:,:,:self.d_l]
        x_a = x[:,:,self.d_l:self.d_l+self.d_a]
        x_v = x[:,:,self.d_l+self.d_a:]

        self.h_l = torch.zeros(n, self.dh_l, dtype=x.dtype).cuda()
        self.h_a = torch.zeros(n, self.dh_a, dtype=x.dtype).cuda()
        self.h_v = torch.zeros(n, self.dh_v, dtype=x.dtype).cuda()
        self.c_l = torch.zeros(n, self.dh_l, dtype=x.dtype).cuda()
        self.c_a = torch.zeros(n, self.dh_a, dtype=x.dtype).cuda()
        self.c_v = torch.zeros(n, self.dh_v, dtype=x.dtype).cuda()
        self.mem = torch.zeros(n, self.mem_dim, dtype=x.dtype).cuda()
        # all_h_ls = []
        # all_h_as = []
        # all_h_vs = []
        # all_c_ls = []
        # all_c_as = []
        # all_c_vs = []
        # all_mems = []
        for i in range(t):
            # prev time step
            prev_c_l = self.c_l
            prev_c_a = self.c_a
            prev_c_v = self.c_v
            # curr time step
            new_h_l, new_c_l = self.lstm_l(x_l[i], (self.h_l, self.c_l))
            new_h_a, new_c_a = self.lstm_a(x_a[i], (self.h_a, self.c_a))
            new_h_v, new_c_v = self.lstm_v(x_v[i], (self.h_v, self.c_v))
            # concatenate
            prev_cs = torch.cat([prev_c_l,prev_c_a,prev_c_v], dim=1)
            new_cs = torch.cat([new_c_l,new_c_a,new_c_v], dim=1)
            cStar = torch.cat([prev_cs,new_cs], dim=1)
            attention = F.softmax(self.att1_fc2(self.att1_dropout(F.relu(self.att1_fc1(cStar)))),dim=1)
            attended = attention*cStar
            cHat = F.tanh(self.att2_fc2(self.att2_dropout(F.relu(self.att2_fc1(attended)))))
            both = torch.cat([attended,self.mem], dim=1)
            gamma1 = F.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
            gamma2 = F.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))
            self.mem = gamma1*self.mem + gamma2*cHat
            # all_mems.append(self.mem)
            # update
            self.h_l, self.c_l = new_h_l, new_c_l
            self.h_a, self.c_a = new_h_a, new_c_a
            self.h_v, self.c_v = new_h_v, new_c_v
            # Detach if max_seqlen is reached to prevent too-deep backpropagation
            if self.max_seqlen is not None and (i + 1) % self.max_seqlen == 0:
                self.h_l = self.h_l.detach()
                self.h_a = self.h_a.detach()
                self.h_v = self.h_v.detach()
                self.c_l = self.c_l.detach()
                self.c_a = self.c_a.detach()
                self.c_v = self.c_v.detach()
                self.mem = self.mem.detach()
            # all_h_ls.append(self.h_l)
            # all_h_as.append(self.h_a)
            # all_h_vs.append(self.h_v)
            # all_c_ls.append(self.c_l)
            # all_c_as.append(self.c_a)
            # all_c_vs.append(self.c_v)

        # last hidden layer last_hs is n x h
        last_h_l = self.h_l
        last_h_a = self.h_a
        last_h_v = self.h_v
        last_mem = self.mem
        last_hs = torch.cat([last_h_l,last_h_a,last_h_v,last_mem], dim=1)
        output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(last_hs))))
        return output
    
if __name__ == "__main__":
    model = MFN3().cuda()
    dataloader = DataLoader(MOSEIDataset("tensors_short.pkl", "train"), batch_size=32)
    for item in dataloader:
        # X's from the dataset is batch * seq_len * embed_dim
        _, X_openface, _, X_glove, X_COAV, y = item
        X_openface = X_openface.transpose(0, 1)
        X_glove = X_glove.transpose(0, 1)
        X_COAV = X_COAV.transpose(0, 1)
        out = model(torch.cat((X_openface, X_glove, X_COAV), dim=2).cuda())
        print(out.shape)
