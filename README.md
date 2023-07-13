# [PNA](https://ojs.aaai.org/index.php/AAAI/article/view/25262/25034) in Detectron2

"[PNA Progressive Neighborhood Aggregation for Semantic Segmentation Refinement](https://ojs.aaai.org/index.php/AAAI/article/view/25262/25034)"

In this branch, we provide the implemented based on Detectron2, and the code based on MMsegmentation is coming soon ...

## Installation
1. Install Detectron2 following [the instructions](https://detectron2.readthedocs.io/tutorials/install.html).

2. Install NATTEN, [Neighborhood Attention Extension](https://github.com/SHI-Labs/NATTEN)  

```bash
cd NATTEN
pip install -e .
```
## Prepare the datasets
```bash
 ln -s your_data_dir ./datasets
``` 
## Download trained models from [google drive](https://drive.google.com/drive/folders/1V5dKfoylfk_wX-UWVst6Ta5ci84c-Kbo?usp=sharing)


## Training &  Testing & Visualizing

#### We have provided the corresponding scripts, just modify it accordingly. 

<strong>To train a model with 4 GPUs run: </strong>
```bash
sh train.sh
``` 
<strong> Testing</strong>
```bash
sh test.sh
``` 
<strong> Visualizing </strong>
```bash
sh demo.sh
```
 
## <a name="CitingPNA"></a>Citing PNA

If you use PNA, please use the following BibTeX entry.

*   PNA:

```
@inproceedings{liu2023progressive,
  title={Progressive Neighborhood Aggregation for Semantic Segmentation Refinement},
  author={Liu, Ting and Wei, Yunchao and Zhang, Yanning},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={2},
  pages={1737--1745},
  year={2023}
} 
```
 
