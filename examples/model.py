"""Demonstrate initialized whole-network force training and inference; no accuracy claim."""
import argparse
import json

import torch
from mace_ictc.hp import create_model,configure_model,energy_and_forces


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--channels",type=int,default=64)
    p.add_argument("--lmax",type=int,default=1)
    p.add_argument("--edge-lmax",type=int)
    p.add_argument("--correlation",type=int,choices=(1,2,3),default=3)
    p.add_argument("--layers",type=int,default=3)
    p.add_argument("--layout",choices=("packed","fused_pack"),default="packed")
    a = p.parse_args()
    torch.manual_seed(73)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model = create_model(channels=a.channels,lmax=a.lmax,edge_lmax=a.edge_lmax,
                         correlation=a.correlation,num_interactions=a.layers,cutoff=4.5)
    n = 64
    pos = torch.randn(n,3,device="cuda",dtype=torch.float32)*.4
    Z = torch.tensor([1,6,8],device="cuda")[torch.arange(n,device="cuda")%3]
    batch = torch.zeros(n,dtype=torch.int64,device="cuda")
    src = torch.arange(n,device="cuda").repeat_interleave(4)
    dst = (src+torch.arange(1,5,device="cuda").repeat(n))%n
    shifts = pos.new_zeros(len(src),3)
    cell = torch.eye(3,device="cuda",dtype=torch.float32)[None]*20
    graph = (pos,Z,batch,src,dst,shifts,cell)
    info = configure_model(model,training=True,policy="generated",layout=a.layout)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    out = energy_and_forces(model,*graph,training=True)
    loss = out["energy"].square().mean()+out["forces"].square().mean()
    loss.backward(inputs=params)
    optimizer.step()
    model.eval().requires_grad_(False)
    configure_model(model,policy="generated",layout=a.layout)
    with torch.no_grad():out = energy_and_forces(model,*graph)
    print(json.dumps({"plan":info["selected"],"layout":info["layout"],"channels":a.channels,
        "lmax":a.lmax,"correlation":a.correlation,"layers":a.layers,
        "loss":float(loss.detach()),"energy_shape":list(out["energy"].shape),
        "force_shape":list(out["forces"].shape)}))


if __name__=="__main__":main()
