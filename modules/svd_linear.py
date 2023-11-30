import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SVDLinear(nn.Module):
    def __init__(
        self, U,S,V, bias=None
    ) -> None:
        super().__init__()
        self.ALinear=nn.Linear(U.size(1), U.size(0), bias=False)
        self.ALinear.weight.data=U.mul(S.sqrt())
        self.BLinear=nn.Linear(V.size(1), V.size(0), bias=False)
        self.BLinear.weight.data=V.t().mul(S.sqrt().view(-1,1))
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None


    @staticmethod
    def from_linear(
        linear: nn.Linear,
        param_ratio: float,
        act_aware=False,
        reorder=False,
        gradient_aware=False,
        ic_split=1,
        oc_split=1,
        train_scale=False,
        act_full=False,
        alpha=1,
    ):
        if param_ratio>=1:
            return linear
        n_params = linear.weight.numel()
        compressed_params = int(n_params * param_ratio)
        assert ic_split==1 or oc_split==1
        if ic_split>1:
            assert linear.in_features%ic_split==0
            rank=compressed_params//(ic_split*linear.out_features+linear.in_features)
        elif oc_split>1:
            assert linear.out_features%oc_split==0
            rank=compressed_params//(oc_split*linear.in_features+linear.out_features)
        else:
            rank = compressed_params // (linear.in_features + linear.out_features)
        # print("rank", rank)
        w=linear.weight.data.float()
        if gradient_aware:
            if linear.in_features>linear.out_features:
                input_g_mean=linear.input_grad.abs()
                # input_g_mean=input_g_mean*linear.weight.data.abs().mean(0) # shape ic
                # input_g_mean=linear.output_grad.abs()
                # input_g_mean=g.abs().mean(0) # shape ic
                input_g_mean+=1e-6 # avoid zero division
                input_g_mean=input_g_mean/input_g_mean.mean() #normalize
                input_g_mean=input_g_mean.sqrt()
                output_g_mean=torch.ones_like(linear.output_grad)
            else:
                output_g_mean=linear.output_grad.abs()
                output_g_mean+=1e-6
                output_g_mean=output_g_mean/output_g_mean.mean()
                output_g_mean=output_g_mean.sqrt()
                input_g_mean=torch.ones_like(linear.input_grad)

            # breakpoint()
            # input_g_mean=torch.log2(input_g_mean).clamp_(min=1e-6)
            w = w*input_g_mean.view(1,-1)
            w = w*output_g_mean.view(-1,1)
        if act_full:
            act_full_mat=linear.full_input.to(w.device).to(w.dtype).view(-1,w.size(1))[:w.size(1)].T
            act_full_mat=torch.where(act_full_mat.abs()<1e-6, 1e-6, act_full_mat)
            try:
                full_input_inv=torch.inverse(act_full_mat.float())
            except:
                print(f"act_full_mat cannot be inversed for {linear}, disable act_full")
                act_full=False
            if act_full:
                w= torch.matmul(w,act_full_mat)
        if act_aware:
            scaling_diag_matrix = linear.scaling_diag_matrix**alpha
            scaling_diag_matrix += 1e-6  # avoid zero division
            w = w * scaling_diag_matrix.view(1, -1)
        if train_scale:
            w = w * F.sigmoid(linear.Si)
            w = w * F.sigmoid(linear.So)
        ic_indexes=None
        oc_indexes=None
        if reorder and max(ic_split,oc_split)>1:
            # deprecated
            if ic_split>1:
                indexes = torch.argsort(linear.scaling_diag_matrix)
                indexes = indexes.view(-1, max(ic_split,oc_split))
                ic_indexes=indexes.transpose(0, 1).reshape(-1)
            if oc_split>1:
                indexes = torch.argsort(linear.output_abs_mean)
                indexes = indexes.view(-1, max(ic_split,oc_split))
                oc_indexes=indexes.transpose(0, 1).reshape(-1)
        if ic_split>1:
            if reorder and max(ic_split,oc_split)>1:
                w=w[:,ic_indexes]
                if act_aware:
                    scaling_diag_matrix=scaling_diag_matrix[ic_indexes]
            w=w.view(linear.out_features, ic_split, linear.in_features//ic_split)
            
            Us=[]
            Ss=[]
            Vs=[]
            for i in range(ic_split):
                try:
                    U, S, V = torch.svd_lowrank(w[:,i,:], q=rank)
                except:
                    print(f"svd failed for {linear}, disable act_aware")
                    return nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                if act_aware:
                    V=V/scaling_diag_matrix.view(ic_split, linear.in_features//ic_split,1)[i]
                if train_scale:
                    V = V / F.sigmoid(linear.Si.view(oc_split,-1, 1)[i])
                    U = U / F.sigmoid(linear.So.view(-1,1))
                Us.append(U)
                Ss.append(S)
                Vs.append(V)
                
            split='ic'
        elif oc_split>1:
            if reorder and max(ic_split,oc_split)>1:
                w=w[oc_indexes]
            w=w.view(oc_split, linear.out_features//oc_split, linear.in_features)
            Us=[]
            Ss=[]
            Vs=[]
            for i in range(oc_split):
                try:
                    U, S, V = torch.svd_lowrank(w[i,:,:], q=rank)
                except:
                    print(f"svd failed for {linear}, disable act_aware")
                    return nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                if act_aware:
                    V=V/scaling_diag_matrix.view(-1,1)
                if train_scale:
                    V = V / F.sigmoid(linear.Si.view(-1, 1))
                    U = U / F.sigmoid(linear.So.view(oc_split,-1,1)[i])
                
                Us.append(U)
                Ss.append(S)
                Vs.append(V)
            split='oc'
        else:
            # use numpy to solve SVD
            # U, S, V = np.linalg.svd(w.cpu().numpy(), full_matrices=False)
            # U = torch.from_numpy(U[:, :rank])
            # S = torch.from_numpy(S[:rank])
            # V = torch.from_numpy(V[:rank, :])
            try:
                U, S, V = torch.svd_lowrank(w, q=rank)
            except:
                print(f"svd failed for {linear}, disable act_aware")
                return nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
            # print(S)
            if gradient_aware:
                V = V / input_g_mean.view(-1, 1)
                U = U / output_g_mean.view(-1,1)
            if act_aware:
                V = V / scaling_diag_matrix.view(-1, 1)
            if train_scale:
                V = V / F.sigmoid(linear.Si.view(-1, 1))
                U = U / F.sigmoid(linear.So.view(-1,1))
            if act_full:
                V=torch.matmul(V.T,full_input_inv).T
            Us=[U]
            Ss=[S]
            Vs=[V]
            split='no'

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None

        # nan check
        for S in Ss:
            if torch.isnan(S).any():
                print("nan in S")
                return nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
        for U in Us:
            if torch.isnan(U).any():
                print("nan in U")
                return nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
        for V in Vs:
            if torch.isnan(V).any():
                print("nan in V")
                return nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)

        assert len(Us)==len(Ss)==len(Vs)==1
        new_linear=SVDLinear(Us[0], Ss[0], Vs[0], bias)
        return new_linear.to(linear.weight.dtype)

    def forward(self, inp):
        # compute USV^Tx + b
        y = self.BLinear(inp)
        y = self.ALinear(y)
            
        if self.bias is not None:
            y = y + self.bias
        return y

