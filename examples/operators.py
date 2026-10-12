"""Run on a qualified remote CUDA host, with task-local caches configured."""
import argparse
import json

import torch
from mace_ictc.hp import EdgeTensorProduct, GraphMessage, direction_harmonics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels",type=int,default=64)
    parser.add_argument("--lmax",type=int,default=1)
    parser.add_argument("--edges",type=int,default=257)
    args = parser.parse_args()
    torch.manual_seed(71)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    op = EdgeTensorProduct.create(channels=args.channels,input_lmax=args.lmax,edge_lmax=args.lmax)
    x = {l:torch.randn(args.edges,args.channels,2*l+1,device="cuda",dtype=torch.float32,requires_grad=True)
         for l in range(args.lmax+1)}
    vectors = torch.randn(args.edges,3,device="cuda",dtype=torch.float32,requires_grad=True)
    Y = direction_harmonics(torch.nn.functional.normalize(vectors,dim=-1),args.lmax,policy="generated")
    gates = torch.randn(args.edges,op.gate_width,device="cuda",dtype=torch.float32,requires_grad=True)
    value = op(op.pack_input(x),torch.cat(Y,-1),gates)
    value.square().mean().backward()
    print(json.dumps({"edge_output_shape":list(value.shape),"paths":len(op.paths),
                      "gate_gradient":gates.grad is not None,"geometry_gradient":vectors.grad is not None}))
    n = 7
    node = {l:torch.randn(n,args.channels,2*l+1,device="cuda",dtype=torch.float32) for l in range(args.lmax+1)}
    src = torch.arange(args.edges,device="cuda")%n
    dst = (src*3+1)%n
    message = GraphMessage(op.tp,input_lmax=args.lmax,edge_lmax=args.lmax)
    output = message(node,{l:y.detach().unsqueeze(1) for l,y in enumerate(Y)},gates.detach(),src,dst)
    print(json.dumps({"node_output_shapes":{l:list(v.shape) for l,v in output.items()}}))


if __name__=="__main__":main()
