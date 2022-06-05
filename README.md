## About
PyTorch code for our submission: **"Logit Margin Matters: Improving Transferable Targeted Adversarial Attack by Logit Calibration"**

The code is implemented based on the Code of the paper [**"On Success and Simplicity: A Second Look at Transferable Targeted Attacks"**](http://arxiv.org/abs/2012.11207).
<br> Zhengyu Zhao, Zhuoran Liu, Martha Larson. **NeurIPS 2021**.

### Requirements
torch>=1.7.0; torchvision>=0.8.1; tqdm>=4.31.1; pillow>=7.0.0; matplotlib>=3.2.2;  numpy>=1.18.1; 

### Dataset

The 1000 images from the NIPS 2017 ImageNet-Compatible dataset are provided in the folder ```dataset/images```, along with their metadata in  ```dataset/images.csv```. More details about this dataset can be found in its [official repository](https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset).

### Evaluation
Following the setting in Zhao et al. NeurIPS 2021, all attacks are integrated with TI, MI, and DI, and run with 300 iterations to ensure convergence, and L<sub>&infin;</sub>=16.

#### ```eval_single_ce05.py```: Temperature-based Calibration.


#### ```eval_single_margin.py```: Margin-based Calibration.


#### ```eval_single_angle.py```: Angle-based Calibration. 
