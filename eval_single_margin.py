"""

evaluate simple transferabl attacks in the single-model transfer setting.

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image
from tqdm import tqdm, tqdm_notebook
import csv
import numpy as np
import os
import scipy.stats as st


##load image metadata (Image_ID, true label, and target label)
def load_ground_truth(csv_filename):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append(row['ImageId'])
            label_ori_list.append(int(row['TrueLabel']) - 1)
            label_tar_list.append(int(row['TargetClass']) - 1)

    return image_id_list, label_ori_list, label_tar_list


## simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


##define TI
def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


channels = 3
kernel_size = 5
kernel = gkern(kernel_size, 3).astype(np.float32)
gaussian_kernel = np.stack([kernel, kernel, kernel])
gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()


##define DI
def DI(X_in):
    rnd = np.random.randint(299, 330, size=1)[0]
    h_rem = 330 - rnd
    w_rem = 330 - rnd
    pad_top = np.random.randint(0, h_rem, size=1)[0]
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem, size=1)[0]
    pad_right = w_rem - pad_left

    c = np.random.rand(1)
    if c <= 0.7:
        X_out = F.pad(F.interpolate(X_in, size=(rnd, rnd)), (pad_left, pad_top, pad_right, pad_bottom), mode='constant',
                      value=0)
        return X_out
    else:
        return X_in


## define Po+Trip
def Poincare_dis(a, b):
    L2_a = torch.sum(torch.square(a), 1)
    L2_b = torch.sum(torch.square(b), 1)

    theta = 2 * torch.sum(torch.square(a - b), 1) / ((1 - L2_a) * (1 - L2_b))
    distance = torch.mean(torch.acosh(1.0 + theta))
    return distance


def Cos_dis(a, b):
    a_b = torch.abs(torch.sum(torch.multiply(a, b), 1))
    L2_a = torch.sum(torch.square(a), 1)
    L2_b = torch.sum(torch.square(b), 1)
    distance = torch.mean(a_b / torch.sqrt(L2_a * L2_b))
    return distance


model_1 = models.inception_v3(pretrained=True, transform_input=False).eval()
model_2 = models.resnet50(pretrained=True).eval()
model_3 = models.densenet121(pretrained=True).eval()
model_4 = models.vgg16_bn(pretrained=True).eval()


for param in model_1.parameters():
    param.requires_grad = False
for param in model_2.parameters():
    param.requires_grad = False
for param in model_3.parameters():
    param.requires_grad = False
for param in model_4.parameters():
    param.requires_grad = False

device = torch.device("cuda:0")
model_1.to(device)
model_2.to(device)
model_3.to(device)
model_4.to(device)


torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# values are standard normalization for ImageNet images,
# from https://github.com/pytorch/examples/blob/master/imagenet/main.py
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trn = transforms.Compose([transforms.ToTensor(), ])
image_id_list, label_ori_list, label_tar_list = load_ground_truth('./dataset/images.csv')

batch_size = 20
max_iterations = 300
input_path = './dataset/images/'
num_batches = np.int32(np.ceil(len(image_id_list) / batch_size))
img_size = 299
lr = 2 / 255  # step size
epsilon = 16  # L_inf norm bound

if not os.path.exists('results_norm'):
    os.mkdir("results_norm")

for iter in range(5):
    # CE_Inc_V3
    pos = np.zeros((3, max_iterations // 20))
    for k in tqdm(range(0, num_batches)):
        batch_size_cur = min(batch_size, len(image_id_list) - k * batch_size)
        X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
        delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
        for i in range(batch_size_cur):
            X_ori[i] = trn(Image.open(input_path + image_id_list[k * batch_size + i] + '.png'))
        labels = torch.tensor(label_tar_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)
        grad_pre = 0
        prev = float('inf')
        for t in range(max_iterations):
            logits = model_1(norm(DI(X_ori + delta)))

            value, _ = torch.sort(logits, dim=1, descending=True)
            logits = logits / torch.unsqueeze(value[:, 0] - value[:, 1], 1).detach()
            loss = nn.CrossEntropyLoss(reduction='sum')(logits, labels)

            loss.backward()
            grad_c = delta.grad.clone()
            grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)  # TI
            grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1 * grad_pre  # MI
            grad_pre = grad_a
            delta.grad.zero_()
            delta.data = delta.data - lr * torch.sign(grad_a)
            delta.data = delta.data.clamp(-epsilon / 255, epsilon / 255)
            delta.data = ((X_ori + delta.data).clamp(0, 1)) - X_ori
            if t % 20 == 19:
                pos[0, t // 20] = pos[0, t // 20] + sum(
                    torch.argmax(model_2(norm(X_ori + delta)), dim=1) == labels).cpu().numpy()
                pos[1, t // 20] = pos[1, t // 20] + sum(
                    torch.argmax(model_3(norm(X_ori + delta)), dim=1) == labels).cpu().numpy()
                pos[2, t // 20] = pos[2, t // 20] + sum(
                    torch.argmax(model_4(norm(X_ori + delta)), dim=1) == labels).cpu().numpy()

    torch.cuda.empty_cache()
    pos_inc_v3_ce = np.copy(pos)
    print("____Inc_V3____")
    print(pos_inc_v3_ce)

    file_name = "results_norm/%d_inc_v3.npy" % iter
    np.save(file_name, pos_inc_v3_ce)

    # CE_ResNet50
    pos = np.zeros((3, max_iterations // 20))
    for k in tqdm(range(0, num_batches)):
        batch_size_cur = min(batch_size, len(image_id_list) - k * batch_size)
        X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
        delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
        for i in range(batch_size_cur):
            X_ori[i] = trn(Image.open(input_path + image_id_list[k * batch_size + i] + '.png'))
        labels = torch.tensor(label_tar_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)
        grad_pre = 0
        prev = float('inf')
        for t in range(max_iterations):
            logits = model_2(norm(DI(X_ori + delta)))
            value, _ = torch.sort(logits, dim=1, descending=True)
            logits = logits / torch.unsqueeze(value[:, 0] - value[:, 1], 1).detach()
            loss = nn.CrossEntropyLoss(reduction='sum')(logits, labels)

            loss.backward()
            grad_c = delta.grad.clone()
            grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)  # TI
            grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1 * grad_pre  # MI
            grad_pre = grad_a
            delta.grad.zero_()
            delta.data = delta.data - lr * torch.sign(grad_a)
            delta.data = delta.data.clamp(-epsilon / 255, epsilon / 255)
            delta.data = ((X_ori + delta.data).clamp(0, 1)) - X_ori
            if t % 20 == 19:
                pos[0, t // 20] = pos[0, t // 20] + sum(
                    torch.argmax(model_1(norm(X_ori + delta)), dim=1) == labels).cpu().numpy()
                pos[1, t // 20] = pos[1, t // 20] + sum(
                    torch.argmax(model_3(norm(X_ori + delta)), dim=1) == labels).cpu().numpy()
                pos[2, t // 20] = pos[2, t // 20] + sum(
                    torch.argmax(model_4(norm(X_ori + delta)), dim=1) == labels).cpu().numpy()

    torch.cuda.empty_cache()
    pos_res50_ce = np.copy(pos)
    print("____ResNet50____")
    print(pos_res50_ce)

    file_name = "results_norm/%d_resnet50.npy" % iter
    np.save(file_name, pos_res50_ce)

    # CE_DenseNet121
    pos = np.zeros((3, max_iterations // 20))
    for k in tqdm(range(0, num_batches)):
        batch_size_cur = min(batch_size, len(image_id_list) - k * batch_size)
        X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
        delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
        for i in range(batch_size_cur):
            X_ori[i] = trn(Image.open(input_path + image_id_list[k * batch_size + i] + '.png'))
        labels = torch.tensor(label_tar_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)
        grad_pre = 0
        prev = float('inf')
        for t in range(max_iterations):
            logits = model_3(norm(DI(X_ori + delta)))
            value, _ = torch.sort(logits, dim=1, descending=True)
            logits = logits / torch.unsqueeze(value[:, 0] - value[:, 1], 1).detach()
            loss = nn.CrossEntropyLoss(reduction='sum')(logits, labels)

            loss.backward()
            grad_c = delta.grad.clone()
            grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)  # TI
            grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1 * grad_pre  # MI
            grad_pre = grad_a
            delta.grad.zero_()
            delta.data = delta.data - lr * torch.sign(grad_a)
            delta.data = delta.data.clamp(-epsilon / 255, epsilon / 255)
            delta.data = ((X_ori + delta.data).clamp(0, 1)) - X_ori
            if t % 20 == 19:
                pos[0, t // 20] = pos[0, t // 20] + sum(
                    torch.argmax(model_1(norm(X_ori + delta)), dim=1) == labels).cpu().numpy()
                pos[1, t // 20] = pos[1, t // 20] + sum(
                    torch.argmax(model_2(norm(X_ori + delta)), dim=1) == labels).cpu().numpy()
                pos[2, t // 20] = pos[2, t // 20] + sum(
                    torch.argmax(model_4(norm(X_ori + delta)), dim=1) == labels).cpu().numpy()

    torch.cuda.empty_cache()
    pos_dense121_ce = np.copy(pos)
    print("____DenseNet121____")
    print(pos_dense121_ce)

    file_name = "results_norm/%d_dense121.npy" % iter
    np.save(file_name, pos_dense121_ce)

    # CE_VGG16
    pos = np.zeros((3, max_iterations // 20))
    for k in tqdm(range(0, num_batches)):
        batch_size_cur = min(batch_size, len(image_id_list) - k * batch_size)
        X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
        delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
        for i in range(batch_size_cur):
            X_ori[i] = trn(Image.open(input_path + image_id_list[k * batch_size + i] + '.png'))
        labels = torch.tensor(label_tar_list[k * batch_size:k * batch_size + batch_size_cur]).to(device)
        grad_pre = 0
        prev = float('inf')
        for t in range(max_iterations):
            logits = model_4(norm(DI(X_ori + delta)))
            value, _ = torch.sort(logits, dim=1, descending=True)
            logits = logits / torch.unsqueeze(value[:, 0] - value[:, 1], 1).detach()
            loss = nn.CrossEntropyLoss(reduction='sum')(logits, labels)

            loss.backward()
            grad_c = delta.grad.clone()
            grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)  # TI
            grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1 * grad_pre  # MI
            grad_pre = grad_a
            delta.grad.zero_()
            delta.data = delta.data - lr * torch.sign(grad_a)
            delta.data = delta.data.clamp(-epsilon / 255, epsilon / 255)
            delta.data = ((X_ori + delta.data).clamp(0, 1)) - X_ori
            if t % 20 == 19:
                pos[0, t // 20] = pos[0, t // 20] + sum(
                    torch.argmax(model_1(norm(X_ori + delta)), dim=1) == labels).cpu().numpy()
                pos[1, t // 20] = pos[1, t // 20] + sum(
                    torch.argmax(model_2(norm(X_ori + delta)), dim=1) == labels).cpu().numpy()
                pos[2, t // 20] = pos[2, t // 20] + sum(
                    torch.argmax(model_3(norm(X_ori + delta)), dim=1) == labels).cpu().numpy()

    torch.cuda.empty_cache()
    pos_vgg16_ce = np.copy(pos)
    print("____VGG16____")
    print(pos_vgg16_ce)

    file_name = "results_norm/%d_vgg16.npy" % iter
    np.save(file_name, pos_vgg16_ce)
