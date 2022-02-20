from model.recursive_model import BodyPoseModel
from torch.utils import data
from data.dataloader import CocoTrainDataset
from torch import nn
from torch import optim
from torchvision import transforms
import torch
import numpy as np


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomRotation((-60, 60)),

])
train_dataset = CocoTrainDataset('./COCO dataset', sigma=7, stride=8, thickness=1)
trainloader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = BodyPoseModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, last_epoch=-1)
torch.set_num_threads(8)

if torch.cuda.is_available():
    pass
else:
    print('cuda is not available')

running_loss = 0.0
count = 0
for epoch in range(280):
    state = {'net': net.state_dict(),
             'optimizer': optimizer.state_dict(),
             'epoch': epoch}
    for i, sample in enumerate(trainloader, 0):
        count += 1
        img, heatmap, pafmap = sample
        heatmap = torch.tensor(heatmap)
        heatmap = heatmap.cuda()
        img = torch.tensor(img)
        img = img.to(device)
        pafmap = torch.tensor(pafmap)
        pafmap = pafmap.cuda()
        optimizer.zero_grad()
        out1, out2 = net(img)
        out = np.vstack([out1, out2])
        groundtruth = np.vstack([heatmap, pafmap])
        loss = criterion(out, groundtruth)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('{}th batch loss is {}'.format(i, loss))
    torch.save(obj=state, f='./model/{}_trained_model.pth'.format(epoch))