# [PNA](https://ojs.aaai.org/index.php/AAAI/article/view/25262/25034) in Detectron2

"[PNA Progressive Neighborhood Aggregation for Semantic Segmentation Refinement](https://ojs.aaai.org/index.php/AAAI/article/view/25262/25034)"

In this branch, we provide the code based on [MMsegmentation](https://github.com/open-mmlab/mmsegmentation), the experiments on Pascal VOC and COCOStuff10k are conducted using this code. And we also provide the code based on [Detectron2](https://detectron2.readthedocs.io/tutorials/install.html). 

## Installation
1. Install MMsegmentation

```bash 
cd pna
pip install mmcv-full==1.6.0
pip install -v -e .
```

2. Install NATTEN, [Neighborhood Attention Extension](https://github.com/SHI-Labs/NATTEN)  

```bash
cd NATTEN
pip install -e .
```
## Prepare the datasets
```bash
 ln -s your_data_dir ./data
``` 
## Download trained models from [google drive](https://drive.google.com/drive/folders/1vAVvQIc1IxealP-u31Z9DRFQUBnjWuMp?usp=sharing)


## Training &  Testing

#### We have provided the corresponding scripts, just modify it accordingly. 

<strong>To train a model run: </strong>
```bash
sh train.sh
``` 
<strong> Testing</strong>
```bash
sh test.sh
```  
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
 
