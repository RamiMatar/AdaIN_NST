# AdaIN_NST
Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization by Xun Huang, Serge Belongie

# Neural Style Transfer
Neural Style Transfer (NST) refers to the process of applying the style of one image (style image) to another image (content image) using neural network methods. The paper, 
"A Neural Algorithm of Artistic Style", by Gatys, et. al. (2015) introduced neural networks to image style transfer through an optimization approach that minimizes the 
perceptual feature loss which is calculated with a pretrained convolutional neural network. Feedforward approaches to this problem shortly followed, and allowed style transfer
in real time, yet restricted the set of styles to a finite set. In 2016, Ulyadov, et. al. found that Instance Normalization allowed the network to carry important style information and
 produce much better results in NST than batch normalization. In 2017, Huang, et. al. showed that a non-learnable Adaptive Instance Normalization layer can
 extend feedforward methods to arbitrary styles. 
 For an extended discussion, read the project report, project_report.pdf
 

# In this repository
* Implementation of the AdaIN paper as an iPython Notebook, including the training process and using the NST to make some cool results!
* Python implementation of the AdaIN paper with multi-gpu training support using DDP.
* Pretrained model

# Required packages
* torch, torchvision
* tqdm
* PIL
* matplotlib

# Datasets
You may use any image datasets with this model, I recommend WikiArt for style dataset if you want a wide range of artistic style transfers. MS COCO or similar image classification datasets may be used. For the pretrained model, I used WikiArt for the style images, and MSCOCO for my content image dataset, supplemented with a dataset of album covers (15% of the size of MSCOCO) from 512 album covers dataset.

# How to use:
1. clone this repository.
2. pip install -r requirement.txt
3. for training, consult python trainer.py -h. for testing, consult python eval.py -h


# Some results
<img src="results/butler_1.png">
<img src="results/butler_2.png">
<img src="results/butler_3.png">
<img src="results/eagle_1.png">
<img src="results/riverside_1.png">
<img src="results/riverside_2.png">

# References and acknowledgements
* Gatys, Leon A., Alexander S. Ecker, and Matthias Bethge. "A neural algorithm of artistic style." arXiv preprint arXiv:1508.06576 (2015).
* Huang, Xun, and Serge Belongie. "Arbitrary style transfer in real-time with adaptive instance normalization." Proceedings of the IEEE international * conference on computer vision. 2017.
* Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
* Johnson, Justin, Alexandre Alahi, and Li Fei-Fei. "Perceptual losses for real-time style transfer and super-resolution." Computer Vision–ECCV 2016: * 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14. Springer International Publishing, 2016.
* Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. "Instance normalization: The missing ingredient for fast stylization." arXiv preprint arXiv:1607.08022 (2016).
* This really helpful youtube playlist and github repo by gordicaleksa https://github.com/gordicaleksa/pytorch-neural-style-transfer-johnson https://www.youtube.com/playlist?list=PLBoQnSflObcmbfshq9oNs41vODgXG-608
* The original implementation of the paper in Torch by the authors https://github.com/xunhuang1995/AdaIN-style


