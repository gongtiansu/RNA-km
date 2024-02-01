import os
import sys
import click
import torch
import torch.nn.functional as F
import numpy as np
from model.pretrain_model import PretrainModel as Model


def load_model(path):
    model = Model(pretrain=False)
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    parsed_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("module."):
            k = k[7:]
        parsed_dict[k] = v
    model.load_state_dict(parsed_dict)
    return model

def inference(seq, weight, cuda):
    here = os.path.dirname(__file__)
    model = load_model(os.path.join(os.path.dirname(__file__),weight))
    seq = seq.upper().replace("U", "T")
    seq = "".join([_ if _ in "ACGT" else "N" for _ in seq])
    vocab = np.full(128, -1, dtype=np.uint8)
    vocab[np.array("ATCGX", "c").view(np.uint8)] = np.arange(len("ATCGX"))
    seq = vocab[np.array(seq, "c").view(np.uint8)]
    data = {"src": torch.from_numpy(seq[None]).long()}
    if cuda:
        model = model.cuda()
        data = {k: v.cuda() for k, v in data.items()}
    with torch.no_grad():
        output = model(data)
    return {
            "sequence_representation": output["residue_repr"][0].cpu().numpy(),
            "attention_weights": output["attn_weight"].squeeze(1).cpu().numpy(),
            }


@click.command()
@click.option("-i", "--fasta", help="Input sequence (fasta format)", required=True)
@click.option(
    "-o", "--outdir", help="Output dictionary", default="./")
@click.option("--attn", is_flag=True, default=False)
@click.option("--cuda", is_flag=True, default=True)

def main(fasta, outdir, attn, cuda):
    here = os.path.dirname(__file__)
    weight = "weight/weight.pth"
    task = open(fasta, "r").read().split('\n')
    name, seq = task[0][1:], task[1].strip()
    output = inference(seq, weight, cuda)
    if attn:
        np.save(os.path.join(outdir, name + ".npy"), output)
    else:
        np.save(os.path.join(outdir, name + ".npy"), output["sequence_representation"])
if __name__ == "__main__":
    main()
