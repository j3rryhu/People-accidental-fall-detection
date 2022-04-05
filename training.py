from model.recursive_model import BodyPoseModel
from torch.utils import data
from data.dataloader import CocoTrainDataset
from torch import nn
from torch import optim
from torchvision import transforms
import torch
import numpy as np


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomRotation((-60, 60)),
])
train_dataset = CocoTrainDataset('./COCO dataset', sigma=7, stride=8, thickness=1, transform=transform)
trainloader = data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = BodyPoseModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=5e-6)
torch.set_num_threads(8)

if torch.cuda.is_available():
    pass
else:
    print('cuda is not available')

running_loss = 0.0
count = 0
for epoch in range(280):
    print('------epoch {}------'.format(epoch))
    state = {'net': net.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch}
    for i, sample in enumerate(trainloader, 0):
        count += 1
        img, heatmap, pafmap = sample.values()
        heatmap = torch.tensor(heatmap)
        heatmap = heatmap.to(device)
        img = img.to(device)
        pafmap = torch.tensor(pafmap)
        pafmap = pafmap.to(device)
        optimizer.zero_grad()
        out1, out2 = net(img)
        out = torch.cat([out1, out2], 1)
        groundtruth = torch.cat([heatmap, pafmap], 1)
        groundtruth = groundtruth.float()
        loss = criterion(out, groundtruth)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('{}th batch loss is {}'.format(i, loss))
    torch.save(obj=state, f='./model/{}_trained_model.pth'.format(epoch))