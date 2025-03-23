import torch
import torch.nn as nn
import torch.nn.functional as F
from mmsdk.mmmodelsdk.fusion.dynamic_fusion_graph.model import DynamicFusionGraph
from mfn import MFN3

class GraphMFN3(MFN3):

    def __init__(self, max_seqlen=None):
        super().__init__(max_seqlen=max_seqlen)
        pattern_model=nn.Sequential(nn.Linear(128, self.mem_dim))
        efficacy_model=nn.Sequential(nn.Linear(128, self.mem_dim))
        self.dfg = DynamicFusionGraph(
            pattern_model,
            [self.dh_l * 2, self.dh_a * 2, self.dh_v * 2],
            self.mem_dim,
            efficacy_model
        )

        self.gamma1_fc1 = nn.Linear(self.mem_dim, self.gamma1_fc1.out_features)
        self.gamma2_fc1 = nn.Linear(self.mem_dim, self.gamma1_fc2.out_features)
    
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
            # prev_cs = torch.cat([prev_c_l,prev_c_a,prev_c_v], dim=1)
            # new_cs = torch.cat([new_c_l,new_c_a,new_c_v], dim=1)
            # cStar = torch.cat([prev_cs,new_cs], dim=1)
            # attention = F.softmax(self.att1_fc2(self.att1_dropout(F.relu(self.att1_fc1(cStar)))),dim=1)
            # attended = attention*cStar
            # cHat = F.tanh(self.att2_fc2(self.att2_dropout(F.relu(self.att2_fc1(attended)))))
            # both = torch.cat([attended,self.mem], dim=1)
            window_l = torch.cat([prev_c_l, new_c_l], dim=1)
            window_a = torch.cat([prev_c_a, new_c_a], dim=1)
            window_v = torch.cat([prev_c_v, new_c_v], dim=1)
            dfg_out, hidden_updates, _ = self.dfg.fusion([window_l, window_a, window_v])
            gamma1 = F.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(dfg_out)))))
            gamma2 = F.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(dfg_out)))))
            self.mem = gamma1 * self.mem + gamma2 * dfg_out

            # update
            # self.h_l, self.c_l = new_h_l + hidden_updates[(0,)][:, :self.dh_l], new_c_l + hidden_updates[(0,)][:, self.dh_l:]
            # self.h_a, self.c_a = new_h_a + hidden_updates[(1,)][:, :self.dh_a], new_c_a + hidden_updates[(1,)][:, self.dh_a:]
            # self.h_v, self.c_v = new_h_v + hidden_updates[(2,)][:, :self.dh_v], new_c_v + hidden_updates[(2,)][:, self.dh_v:]
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

        # last hidden layer last_hs is n x h
        last_h_l = self.h_l
        last_h_a = self.h_a
        last_h_v = self.h_v
        last_mem = self.mem
        last_hs = torch.cat([last_h_l,last_h_a,last_h_v,last_mem], dim=1)
        output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(last_hs))))
        return output
