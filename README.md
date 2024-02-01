# RNA-km
## About The Project

The implementation of the paper "Language models enable zero-shot prediction of RNA secondary structure including pseudoknots".

## Getting Started
### Prerequisites
Install [PyTorch 1.6+](https://pytorch.org/),
[python
3.7+](https://www.python.org/downloads/)

### Installation

1. Clone the repo
```sh
git clone https://github.com/gongtiansu/RNA-km.git
```

2. Install python packages
```sh
cd RNA-km
pip install -r requirements.txt
```
3. Download pretrained [model weight](https://drive.google.com/file/d/1DI79-R33R396ZdbNcqbADQMP7Nprb_ze/view?usp=sharing) and place the pth file into the weight folder

## Usage
1. RNA-km.py: extract RNA sequence representation (L * 1024) and attention maps from sequence (fasta format)  
```sh
python RNA-km.py -i <RNA_fasta> -o <output_dictionary> (--attn) (--cuda)
```
## Example
```sh
cd example
./run_example.sh
```

## License
Distributed under the MIT License. See `LICENSE` for more information.
