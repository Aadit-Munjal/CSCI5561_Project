'''Implementation of semantic segmentation based on fully convolutional network (FCN) from scratch.'''

import os
import torch
import torchvision
import toolbox as ricky
import torchvision
from torch import nn
import numpy as np

#voc_dir = ricky.download_extract('voc2012','VOCdevkit/VOC2012')

def read_voc_images(voc_dir, is_train=True):
    """Read all VOC features and lobel images."""
    txt_fname = os.path.join(voc_dir,'ImageSets','Segmentation','train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

#train_features, train_labels = read_voc_images(voc_dir, True)


def test_image():
    n = 5
    imgs = train_features[:n] + train_labels[:n]
    imgs = [img.permute(1,2,0) for img in imgs]
    ricky.show_images(imgs, 2, n)
    ricky.plt.show()

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def voc_colormap2label():
    """Build the mapping from RGB to class indices for VOC labels."""
    colormap2label = torch.zeros(256**3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0]*256+colormap[1])*256+colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    """Map any RGB values in VOC labels to their class indices."""
    colormap = colormap.permute(1,2,0).numpy().astype('int32')
    idx = ((colormap[:,:,0]*256+colormap[:,:,1])*256+colormap[:,:,2])
    return colormap2label[idx]

def test_label_indices():
    y = voc_label_indices(train_labels[0], voc_colormap2label())
    print(y[105:115, 130:140], VOC_CLASSES[1])

def voc_rand_crop(feature, label, height, width):
    """Randomly crop both feature and label images."""
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height,width))
    feature = torchvision.transforms.functional.crop(feature,*rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

def testtest():
    n = 5
    imgs = []
    for _ in range(n):
        imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 200)
    imgs = [img.permute(1,2,0) for img in imgs]
    ricky.show_images(imgs[::2] + imgs[1::2], 2, n)
    ricky.plt.show()

class VOCSegDataset(torch.utils.data.Dataset):
    """A customized dataset to load the VOC dataset."""
    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature) for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')
    
    def normalize_image(self, img):
        return self.transform(img.float() / 255)
    
    def filter(self, imgs):
        return [img for img in imgs if (img.shape[1] >= self.crop_size[0] and img.shape[2] >= self.crop_size[1])]
    
    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx], *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))
    
    def __len__(self):
        return len(self.features)

def test_data_load():
    crop_size = (320, 480)
    voc_train = VOCSegDataset(True, crop_size, voc_dir)
    voc_test = VOCSegDataset(False, crop_size, voc_dir)

    batch_size = 64
    train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                        drop_last=True,
                                        num_workers=ricky.get_dataloader_workers())
    for X, Y in train_iter:
        print(X.shape)
        print(Y.shape)
        break

def load_data_voc(batch_size, crop_size):
    """Load the VOC semantic segmentation dataset."""
    voc_dir = ricky.download_extract('voc2012', os.path.join('VOCdevkit', 'VOC2012'))
    num_workers = ricky.get_dataloader_workers()
    num_workers = 0
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, test_iter


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight


def loss(inputs, targets):
    return ricky.F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)


def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])


def label2image(pred):
    colormap = torch.tensor(VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X,:]


if __name__ == '__main__':
    pretrained_net = torchvision.models.resnet18(pretrained=True)
    net = nn.Sequential(*list(pretrained_net.children())[:-2])
    num_classes = 21
    net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
    net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                                        kernel_size=64, padding=16, stride=32))
    W = bilinear_kernel(num_classes, num_classes, 64)
    net.transpose_conv.weight.data.copy_(W)
    batch_size, crop_size = 32, (320, 480)
    train_iter, test_iter = load_data_voc(batch_size, crop_size)
    num_epochs, lr, wd, devices = 5, 0.001, 1e-3, ricky.try_all_gpus()
    trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
    ricky.train_gpu(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
    
    voc_dir = ricky.download_extract('voc2012', 'VOCdevkit/VOC2012')
    test_images, test_labels = read_voc_images(voc_dir, False)
    n, imgs = 4, []
    for i in range(n):
        crop_rect = (0, 0, 320, 480)
        X = torchvision.transforms.functional.crop(test_images[i+41], *crop_rect)
        pred = label2image(predict(X))
        imgs += [X.permute(1,2,0), pred.cpu(),
                torchvision.transforms.functional.crop(
                    test_labels[i+41], *crop_rect).permute(1,2,0)]
    ricky.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)
    ricky.plt.show()