# FSDA: FreScaling in Data Augmentation for Corruption-Robust Image Classification
## Abstract
Modern Convolutional Neural Networks (CNNs) are used in various artificial intelligence applications, including computer vision, speech recognition, and robotics. However, practical usage in various applications requires large-scale datasets, and real-world data contains various corruptions, which degrade the model’s performance due to inconsistencies in train and test distributions. In this study, we propose Frequency re-Scaling Data Augmentation (FSDA) to improve the classification performance, robustness against corruption, and localizability of a classifier trained on various image classification datasets. Our method is designed with two processes; a mask generation process and a pattern re-scaling process. Each process clusters spectra in the frequency domain to obtain similar frequency patterns and scale frequency by learning re-scaling parameters from frequency patterns. Since CNN classifies images by paying attention to their structural features highlighted with FSDA, CNN trained with our method has more robustness against corruption than other data augmentations (DAs). Our method achieves higher performance on three public image classification datasets such as CIFAR-10/100 and STL-10 than other DAs. In particular, our method significantly improves robustness against various corruption error by 5.47% over baseline on average and the localizability of the classifier.

![ours (1)](https://user-images.githubusercontent.com/127758215/224719744-602afaf6-0504-4ace-9fa9-f19f0f348a77.png)
![Screenshot from 2023-03-13 22-44-58](https://user-images.githubusercontent.com/127758215/224720104-ca411336-1c5e-4f15-aa1f-c362b38502d3.png)

## Experiment Results
### CIFAR-10 and CIFAR-100 Image Classification Results
![Screenshot from 2023-03-13 22-45-42](https://user-images.githubusercontent.com/127758215/224720627-71d3a4af-9ad6-4eb5-891d-478807c6fc0a.png)
