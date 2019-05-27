import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
import json
import os
import cv2
from PIL import Image
import time

frame_count = 3

class SSV2Dataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        super(SSV2Dataset, self).__init__()
        self.transform = transform
        
        data_jsonfile = os.path.join(root, f'something-something-v2-{mode}.json')
        if not os.path.exists(data_jsonfile):
            print('{} is not exist.'.format(data_jsonfile))

        with open(data_jsonfile, 'r', encoding='utf-8') as f:
            self.data_dict = json.load(f)
            frames_file = os.path.join(root, f'something-something-v2-opencv_{mode}_frames.json')
            with open(frames_file, 'r', encoding='utf-8') as ff:
                frames_dict = json.load(ff)
            if mode != 'test':
                label_jsonfile = os.path.join(root, 'something-something-v2-labels.json')
                if not os.path.exists(label_jsonfile):
                    print('{} is not exist.'.format(label_jsonfile))
                with open(label_jsonfile, 'r', encoding='utf-8') as ftest:
                    labels = json.load(ftest)
            for item in self.data_dict:
                item['frames_total'] = frames_dict[item['id']]
                item['id'] = os.path.join(root, '20bn-something-something-v2', item['id'] + '.webm')
                if mode != 'test':
                    item['classidx'] = int(labels[item['template'].replace('[', '').replace(']', '')])

    def loader(self, path, total_frames):
        vc = cv2.VideoCapture(path)
        index = np.linspace(0, total_frames, frame_count + 1, dtype=np.long)
        count = 0
        images = []
        if vc.isOpened(): #判断是否正常打开
            rval , frame = vc.read()
        else:
            rval = False
        while rval:
            images.append(self.transform(Image.fromarray(frame)).numpy())
            count += 1
            vc.set(cv2.CAP_PROP_POS_FRAMES, index[count])
            rval , frame = vc.read()
        # if len(images) != 10:
        #     print(path, total)
        # index = np.linspace(0, len(images), frame_count + 2, dtype=np.long)
        vc.release()
        return np.array(images)#[index[1:-1]]
    
    def __getitem__(self, index):
        path, total_frames, target = self.data_dict[index]['id'], self.data_dict[index]['frames_total'], self.data_dict[index]['classidx']
        sample = self.loader(path, total_frames)
        return torch.Tensor(sample), target
    
    def __len__(self):
        return len(self.data_dict)

class SSV1Dataset(Dataset):
    def __init__(self, root, mode='train', transform=None):
        super(SSV1Dataset, self).__init__()
        self.transform = transform
        self.root = root
        self.data_dict = np.loadtxt(os.path.join(root, mode +'_videofolder.txt'), dtype=np.long)

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return self.transform(img.convert('RGB')).numpy()

    def loader(self, videoitem):
        path = os.path.join(self.root, '20bn-something-something-v1', str(videoitem[0]))
        index = np.linspace(0, videoitem[1], frame_count + 2, dtype=np.long)
        images = []
        for i in index[1:-1]:
            images.append(self.pil_loader(path + '/{:05d}.jpg'.format(i + 1)))
        return images
    
    def __getitem__(self, index):
        videoitem = self.data_dict[index]
        sample = self.loader(videoitem)
        return torch.Tensor(sample), videoitem[2]
    
    def __len__(self):
        return len(self.data_dict)

batch_size = 32

def collate_fn(batch):
    result, label = [], []
    for i in batch:
        for j in i[0]:
            result.append(j.numpy())
        label.append(i[1])
    return torch.Tensor(result), torch.Tensor(label).long()

train_transform = torchvision.transforms.Compose([
    transforms.Resize((240, 320)),
    transforms.ToTensor(),
])
val_transform = torchvision.transforms.Compose([
    transforms.Resize((240, 320)),
    transforms.ToTensor(),
])
train_set = SSV2Dataset(root='./something-something-v2', mode='train', transform=train_transform)
val_set = SSV2Dataset(root='./something-something-v2', mode='validation', transform=val_transform)

# def collate_fn(batch):
#     result, label = [], []
#     for i in batch:
#         for j in i[0]:
#             result.append(j.numpy())
#         label.append(i[1])
#     return torch.Tensor(result), torch.Tensor(label).long()

# train_transform = torchvision.transforms.Compose([
#     transforms.Resize((100, 224)),
#     transforms.ToTensor(),
# ])
# val_transform = torchvision.transforms.Compose([
#     transforms.Resize((100, 224)),
#     transforms.ToTensor(),
# ])
# train_set = SSV1Dataset(root='./something-something-v1', mode='train', transform=train_transform)
# val_set = SSV1Dataset(root='./something-something-v1', mode='val', transform=val_transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn = collate_fn)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn = collate_fn)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
 
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
 
        return x

def resnet50(num_classes=1024):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

class BasicModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes=1000):
        super(BasicModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.videoframe = frame_count
        self.resnet50 = resnet50(embedding_dim)
        if torch.cuda.device_count() > 1:
            weight = torch.load('resnet50-19c8e357.pth')
            weight.pop('fc.weight', None)
            weight.pop('fc.bias', None)
            self.resnet50.load_state_dict(weight)
            self.resnet50 = nn.DataParallel(self.resnet50,device_ids=[4,5,6,7])

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, hidden_dim))
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        embeds = self.resnet50(x)
        lstm_out, _ = self.lstm(embeds.view(-1, self.videoframe, self.embedding_dim).transpose(1, 0))
        lstm_out = lstm_out.transpose(1, 0).view(-1, self.videoframe, self.hidden_dim)[:, -1]
        tag_scores = self.fc(lstm_out)
        return tag_scores

def save_model(model, save_path):
    # save model
    torch.save(model.state_dict(), save_path)

def train(model, train_loader, loss_func, optimizer, device):
    total_loss = 0
    correct = 0
    total = 0
    model.train()
    for i, (images, targets) in enumerate(train_loader):
        images, targets = Variable(images.to(device)), Variable(targets.to(device))
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)    # make prediction according to the outputs
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item() # count how many predictions is correct
        
        if (i+1) % 100 == 0:
            # print loss and acc
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + (' | Train AVG loss: %6.3f | Train Acc: %6.3f%% (%d/%d)'
                % (total_loss/(i+1), 100.*correct/total, correct, total)))

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + (' | Train AVG loss: %6.3f | Train Acc: %6.3f%% (%d/%d)'
        % (total_loss/(i+1), 100.*correct/total, correct, total)))
 
    return total_loss / len(train_loader)

def evaluate(model, val_loader, loss_func, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        correct = 0
        total = 0 
        for i, (images, targets) in enumerate(val_loader):
            images, targets = Variable(images.to(device)), Variable(targets.to(device))
            outputs = model(images)
            loss = loss_func(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)    # make prediction according to the outputs
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item() # count how many predictions is correct

            if (i+1) % 100 == 0:
                # print loss and acc
                print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + (' | Test  AVG loss: %6.3f | Test  Acc: %6.3f%% (%d/%d)'
                    % (total_loss/(i+1), 100.*correct/total, correct, total)))

        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + (' | Test  AVG loss: %6.3f | Test  Acc: %6.3f%% (%d/%d)'
            % (total_loss/(i+1), 100.*correct/total, correct, total)))

        return total_loss / len(val_loader)

lr = 1e-2
embedding_dim, hidden_dim = 2048, 4096
n_labels = 174

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
model = BasicModel(embedding_dim, hidden_dim, n_labels)
model.to(device)
# if os.path.exists('model/BasicModel-pre.pt'):
#     model.load_state_dict(torch.load('model/BasicModel-pre.pt'))
loss_func = nn.CrossEntropyLoss().to(device)
# optimizer
# define the SGD optimizer, lr、momentum、weight_decay is adjustable
# optimizer = optim.SGD(model.parameters(), lr=lr)
# define the Adam
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

losses = []
accs = []
num_epochs = 20

if not os.path.exists("model"):
    os.mkdir("model")

for epoch in range(num_epochs):
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + ' | Epoch {}/{}:'.format(epoch + 1, num_epochs))
    # train step
    loss = train(model, train_loader, loss_func, optimizer, device)
    losses.append(loss)
    # evaluate step
    evaluate_loss = evaluate(model, val_loader, loss_func, device)
    accs.append(evaluate_loss)

    save_model(model, "model/BasicModel{}.pt".format(epoch + 1))

    # if (epoch + 1) % 1 == 0 :
    #     lr = 0.8 * lr
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr