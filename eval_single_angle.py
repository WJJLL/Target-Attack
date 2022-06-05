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


if not os.path.exists('results_angle'):
    os.mkdir("results_angle")

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


def forward_incv3(input):
    # N x 3 x 299 x 299
    x = model_1.Conv2d_1a_3x3(input)
    # N x 32 x 149 x 149
    x = model_1.Conv2d_2a_3x3(x)
    # N x 32 x 147 x 147
    x = model_1.Conv2d_2b_3x3(x)
    # N x 64 x 147 x 147
    x = model_1.maxpool1(x)
    # N x 64 x 73 x 73
    x = model_1.Conv2d_3b_1x1(x)
    # N x 80 x 73 x 73
    x = model_1.Conv2d_4a_3x3(x)
    # N x 192 x 71 x 71
    x = model_1.maxpool2(x)
    # N x 192 x 35 x 35
    x = model_1.Mixed_5b(x)
    # N x 256 x 35 x 35
    x = model_1.Mixed_5c(x)
    # N x 288 x 35 x 35
    x = model_1.Mixed_5d(x)
    # N x 288 x 35 x 35
    x = model_1.Mixed_6a(x)
    # N x 768 x 17 x 17
    x = model_1.Mixed_6b(x)
    # N x 768 x 17 x 17
    x = model_1.Mixed_6c(x)
    # N x 768 x 17 x 17
    x = model_1.Mixed_6d(x)
    # N x 768 x 17 x 17
    x = model_1.Mixed_6e(x)
    # N x 768 x 17 x 17
    # N x 768 x 17 x 17
    x = model_1.Mixed_7a(x)
    # N x 1280 x 8 x 8
    x = model_1.Mixed_7b(x)
    # N x 2048 x 8 x 8
    x = model_1.Mixed_7c(x)
    # N x 2048 x 8 x 8
    # Adaptive average pooling
    x = model_1.avgpool(x)
    # N x 2048 x 1 x 1
    x = model_1.dropout(x)
    # N x 2048 x 1 x 1
    x = torch.flatten(x, 1)
    # N x 2048
    logits = model_1.fc(x)
    # N x 1000 (num_classes)
    return x, logits


def forward_resnet50(input):
    x = model_2.conv1(input)
    x = model_2.bn1(x)
    x = model_2.relu(x)
    x = model_2.maxpool(x)

    x = model_2.layer1(x)
    x = model_2.layer2(x)
    x = model_2.layer3(x)
    x = model_2.layer4(x)

    x = model_2.avgpool(x)
    x = torch.flatten(x, 1)
    logits = model_2.fc(x)

    return x, logits


def forward_dense(x):
    features = model_3.features(x)
    out = F.relu(features, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    logits = model_3.classifier(out)
    return out, logits

def forward_vgg16(x):

    # nn.Linear(512 * 7 * 7, 4096),
    # nn.ReLU(True),
    # nn.Dropout(p=dropout),
    # nn.Linear(4096, 4096),
    # nn.ReLU(True),
    # nn.Dropout(p=dropout),
    # nn.Linear(4096, num_classes),

    x = model_4.features(x)
    x = model_4.avgpool(x)
    x = torch.flatten(x, 1)
    x = model_4.classifier[0](x)
    x = model_4.classifier[1](x)
    x = model_4.classifier[2](x)
    x = model_4.classifier[3](x)
    x = model_4.classifier[4](x)
    x = model_4.classifier[5](x)
    logits = model_4.classifier[6](x)

    return x, logits


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

bias_inc = model_1.fc.bias.data
weight_inc = model_1.fc.weight.data

bias_res50 = model_2.fc.bias.data
weight_res50 = model_2.fc.weight.data

bias_dense121 = model_3.classifier.bias.data
weight_dense121 = model_3.classifier.weight.data

bias_vgg16 = model_4.classifier[-1].bias.data
weight_vgg16 = model_4.classifier[-1].weight.data

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
            feature, logits = forward_incv3(norm(DI(X_ori + delta)))
            output = F.linear(F.normalize(feature), F.normalize(weight_inc))

            real = output.gather(1, labels.unsqueeze(1)).squeeze(1)
            logit_dists = (-1 * real)
            loss = logit_dists.sum()

            # loss = nn.CrossEntropyLoss(reduction='sum')(output, labels)

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

    file_name = "results_angle/%d_inc_v3.npy" % iter
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
            feature, logits = forward_resnet50(norm(DI(X_ori + delta)))
            output = F.linear(F.normalize(feature), F.normalize(weight_res50))

            real = output.gather(1, labels.unsqueeze(1)).squeeze(1)
            logit_dists = (-1 * real)
            loss = logit_dists.sum()

            # loss = nn.CrossEntropyLoss(reduction='sum')(output, labels)

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

    file_name = "results_angle/%d_resnet50.npy" % iter
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
            feature, logits = forward_dense(norm(DI(X_ori + delta)))
            output = F.linear(F.normalize(feature), F.normalize(weight_dense121))

            real = output.gather(1, labels.unsqueeze(1)).squeeze(1)
            logit_dists = (-1 * real)
            loss = logit_dists.sum()
            # loss = nn.CrossEntropyLoss(reduction='sum')(output, labels)

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

    file_name = "results_angle/%d_dense121.npy" % iter
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
            feature, logits = forward_vgg16(norm(DI(X_ori + delta)))
            output = F.linear(F.normalize(feature), F.normalize(weight_vgg16))

            real = output.gather(1, labels.unsqueeze(1)).squeeze(1)
            logit_dists = (-1 * real)
            loss = logit_dists.sum()
            # loss = nn.CrossEntropyLoss(reduction='sum')(output, labels)

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

    file_name = "results_angle/%d_vgg16.npy" % iter
    np.save(file_name, pos_vgg16_ce)
