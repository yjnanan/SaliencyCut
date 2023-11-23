# SaliencyCut

Code for ["SaliencyCut: Augmenting Plausible Anomalies for Anomaly Detection"](http://arxiv.org/abs/2306.08366).

# Requirements
* matplotlib==3.5.1  
* numpy==1.21.5  
* pandas==1.3.5  
* Pillow==8.4.0  
* scikit_learn==1.0.2  
* torch==1.9.0  
* torchvision==0.10.0  
* tqdm==4.64.0

# Run
```
python train.py --dataset_root=./data/mvtec_anomaly_detection --classname='carpet' --nAnomaly=10 --know_class='cut'
```
* ```dataset_root``` denotes the path of the dataset.
* ```classname``` denotes the subset name of the dataset.
* ```nAnomaly``` denotes the number of anomaly samples involved in training (general setting: 10, hard setting: 1, anomaly-free setting: 0).
* ```know_class``` (optional) specifies the anomaly category in the training set to evaluate under hard setting.

# Citation
```
@article{ye2023saliencycut,
  title={SaliencyCut: Augmenting Plausible Anomalies for Open-set Fine-Grained Anomaly Detection},
  author={Ye, Jianan and Hu, Yijie and Yang, Xi and Wang, Qiu-Feng and Huang, Chao and Huang, Kaizhu},
  journal={arXiv preprint arXiv:2306.08366},
  year={2023}
}
```
