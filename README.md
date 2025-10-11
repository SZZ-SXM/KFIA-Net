# KFIA-Net

> [KFIA-Net: A Knowledge Fusion and Imbalance-Aware Network for Multi-Category SAR Ship Detection]

<!-- [ALGORITHM] -->

## Abstract
Multi-category SAR ship detection is limited by heterogeneity in imaging mechanisms and severe class imbalance. These factors make the classification branch insensitive to minority classes, yielding accurate localization but frequent misclassification. To address this issue, this paper proposes a Knowledge Fusion and Imbalance-Aware Network (KFIA-Net). Our KFIA-Net injects domain knowledge into the classification branch while keeping the regression branch unaffected, thereby improving classification without harming localization. Specifically, we first propose a Domain Knowledge Feature Extraction (DKFE) to extract and encode ship ROI-level knowledge tokens from four priors: geometry, gradient, scatter, and point cloud. Second, a Scale-aware Knowledge Cross-Attention Fusion (SKCAF) module is designed to perform interpretable and sparsely selectable channel modulation on multi-scale categorical features using cross-attention and FiLM decoding. Furthermore, we further design an Imbalance-Aware Loss Function (IALF) that combines prior calibration, tail-class margin expansion, and knowledge-consistency weighting to reduce loss bias under class imbalance. Finally, systematic experiments and comparisons are conducted on three datasets: SRSDD-v1.0, FAIR-CSAR-v1.0, and NUDT-SARship-v1.0. The results show that KFIA-Net achieves mAP50 scores of 64.29%, 37.99%, and 78.26% on these three datasets, respectively. Even with stricter thresholds, it maintains its lead, achieving mAP75 scores of 34.96%, 19.70%, and 66.36%, respectively. These results demonstrate dual benefits of knowledge injection: more accurate classification and robust localization at high IoU. Furthermore, KFIA-Net requires only 11.47M parameters and 66.79G FLOPs, achieving an inference speed of 47.21 FPS on a 1024×1024 input, achieving a good trade-off between accuracy and efficiency.

## Network overview
Our detector follows a single-stage, multi-scale paradigm and leverages domain knowledge to enhance classification branch features while preserving locality. As shown in Fig. 3, the network follows the conventional object detection network architecture, first extracting multi-scale features, then decoupling regression and classification features, and finally predicting the object category and regression bounding box information. Our designed network primarily incorporates three innovative structures: DKFE, SKCFA, and IALF. In parallel with the backbone network, we extract four types of domain knowledge features: Regional Geometric (RGE) Features, Histogram of Oriented Gradients (HOG) Features, Scattering Center Clustering (SCC) Features, and Scattering Point Cloud (SPC) Features. We group these into three categories: vector-based (RGE, HOG), graph-based (SCC), and point cloud-based (SPC). Each category is encoded into a type-specific learnable encoder to a common token space of dimension   by a specific type of learnable encoder: an MLP tokenizer for vectors, a Conv-BN pooling tokenizer for graphs, and a PointNet-style tokenizer for point clouds. The resulting tokens are concatenated and passed through a subset-gating + mixture-of-experts selection module that activates a sparse, ROI-adaptive subset of knowledge sources. To better integrate the extracted domain knowledge features into conventional classification features without increasing the burden of regression features, we propose the SKCFA module. This design preserves the spatial layout, selectively amplifies knowledge-consistent discriminative channels, and keeps the regression pathway intact. In terms of loss function, we designed the IALF function to further reduce the impact of classification loss caused by class imbalance. 

<div align=center>
<img src="https://github.com/SZZ-SXM/KFIA-Net/tree/main/data/Figure3.png">
</div>

## Experimental Dataset
We conducted extensive experimental analysis to fully validate the effectiveness of our proposed model on three different datasets: SRSDD-v1.0, FAIR-CSAR-v1.0, and NUDT-SARship-v1.0. SRSDD-v1.0 and FAIR-CSAR-v1.0 are two of the most classic SAR ship detection datasets covering multiple categories. NUDT-SARship-v1.0 is a dataset currently under development by our research team and will be published soon for use in multi-category SAR ship detection research. The SRSDD-v1.0 dataset is constructed from 30 1-meter-resolution port panoramas from the Gaofen-3 satellite and tiled into 1024×1024-pixel blocks, covering six types of ships. Through visualization experiments, we found that there were some errors in the data annotation, so we recalibrated the labels of the dataset. The corrected dataset contains 666 image slices and a total of 3783 ships in 6 categories. FAIR-CSAR-v1.0 is a benchmark for fine-grained object detection and recognition in complex-domain SAR imagery. It collects 175 high-resolution Gaofen-3 images from 32 cities and multiple sea areas, annotating over 340,000 instances across five major categories and 22 subcategories. In this experiment, we selected ship images from these datasets to create a 7-category dataset containing 8,581 1024x1024 images of 37,131 ships. NUDT-SARship-v1.0 is a multi-category SAR ship detection dataset we are currently developing. Similar to the previous two datasets, this dataset uses 1m-resolution large-scale scene data from the Gaofen-3 satellite. After cropping and annotating, it contains 14,165 images and 36,411 ships across ten categories. The main features of the three experimental datasets are listed in Table 2. To facilitate statistical comparison of experimental results, we have uniformly abbreviated the labels of the three datasets. Fig. 6 shows the number distribution of each category sample in the three experimental datasets. This sample distribution indicates that current multi-category SAR ship detection datasets all suffer from significant class imbalance problem. To facilitate comparative experiments, we have compiled and open-sourced all three datasets.

All calibration data mentioned above can be accessed via the Baidu Netdisk link: https://pan.baidu.com/s/1Xzf-E18PNP5Bugu2NnTsqQ?pwd=NUDT.

## Visualization result
To more intuitively demonstrate the effectiveness of our method, we plotted visualizations of our method on three different datasets. As shown in Fig. 12, directed boxes of different colours correspond to different ship categories, and the one-to-one colour correspondence is intuitively evident. While maintaining high-quality localization, our method also achieves highly accurate classification. The colour of the predicted box for the same ship is consistent with the GT colour, and it can reliably distinguish adjacent and similar-looking categories even in scenes with densely moored objects, large scale differences, or weakly scattered small targets. In particular, cross-class misclassification is almost non-existent in densely packed harbour segments and sparsely packed open ocean segments. The visualizations demonstrate that injecting domain knowledge features into the classification branch to strengthen class discriminative representation significantly reduces cross-class confusion under complex backgrounds and class imbalance, ultimately achieving an overall detection effect of both precise localization and accurate classification.

<div align=center>
<img src="https://github.com/SZZ-SXM/KFIA-Net/tree/main/data/Figure12.png">
</div>

## Quantitative result
Using the same training rounds and data settings, we divided the comparison methods into four categories: anchor-free methods, DETR-based methods, two-stage methods, and one-stage methods. Table 3 shows that our method, KFIA-Net, achieved a consistent lead on all three datasets. Specifically, on NUDT-SARShip-v1.0, its mAP50, mAP60, and mAP75 reached 78.26%, 76.13%, and 72.02%, respectively, significantly outperforming all representative models. On SRSDD-v1.0, its mAP50 reached 64.29% and its mAP75 reached 34.96%, surpassing the top performer, PKINet (64.05% mAP50) in the two-stage dataset, and the top performer, RTMDet-OBB (33.50% mAP75). On FAIR-CSAR-v1.0, KFIA-Net achieved mAP50, mAP60, and mAP75 scores of 38.00%, 32.86%, and 19.70%, respectively, outperforming strong baselines such as LSKNet. The improvement was particularly pronounced at high IoU thresholds. Based on the above analysis, we achieved either first place or second-best results in multiple categories across the competitive landscape of four major method sets, demonstrating our cross-dataset generalization capabilities. Our continued leadership of mAP75 demonstrates the more accurate localization and pose regression of our method. We also maintained our advantage on FAIR-CSAR, which has a more uneven distribution of classes and scales, demonstrating the significant gains achieved by knowledge injection and imbalanced adaptation mechanisms for complex data. Overall, KFIA-Net outperforms existing methods in accuracy, robustness, and transferability, demonstrating its practical value for complex SAR scenarios.

<div align=center>
<img src="https://github.com/SZZ-SXM/KFIA-Net/tree/main/data/result.png">
</div>

**Note**:
All experiments were conducted on a cloud platform, using an Intel(R) Xeon(R) Platinum 8352V CPU @ 2.10GHz and an NVIDIA GeForce RTX 4090 GPU. We used Python 3.13.5, PyTorch 1.11.0, and CUDA 11.3 for GPU acceleration. For single-stage networks, we used an initial learning rate of 0.0025, while for other networks, we used an initial learning rate of 0.005. We used either an SGD optimizer with a learning momentum of 0.9 and a weight drop of 0.0001, or an AdamW optimizer with a weight drop of 0.05. We trained for no more than 100 epochs across all datasets. Other hyperparameters were optimized through iterative experimentation.

## Contact
Welcome to raise issues or email to sunzhongzhen14@163.com or sunzhongzhen14@nudt.edu.cn for any question regarding our MSDFF-Net.

## Citation

```
@ARTICLE{,
  author={Sun, Zhongzhen and Zhang, Xianghui and Leng, Xiangguang and Wu, Xueqi and Xiong, Boli and Ji, Kefeng and Kuang, Gangyao},
  journal={Under Reviewing}, 
  title={KFIA-Net: A Knowledge Fusion and Imbalance-Aware Network for Multi-Category SAR Ship Detection}, 
  year={2025},
  volume={},
  number={},
  pages={1-21},
  keywords={Multi-category SAR ship detection, knowledge token extraction, cross-attention fusion, imbalance-aware loss},
  doi={...}}
```
