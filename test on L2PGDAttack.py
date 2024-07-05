'''
Author: 冯明 10449281+weiyang-v@user.noreply.gitee.com
Date: 2024-07-02 11:03:47
LastEditors: 冯明 10449281+weiyang-v@user.noreply.gitee.com
LastEditTime: 2024-07-03 10:43:12
FilePath: \Pytorch-Adversarial-Training-CIFAR-master\test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from models import *
from advertorch.attacks import L2PGDAttack

file_name = 'basic_training'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

net = ResNet18()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

# Load the checkpoint and map to CPU if CUDA is not available
checkpoint_path = './checkpoint/pgd_adversarial_training'
if device == 'cpu':
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
else:
    checkpoint = torch.load(checkpoint_path)
net.load_state_dict(checkpoint['net'])

adversary = L2PGDAttack(
    net, loss_fn=nn.CrossEntropyLoss(), eps=1.0, nb_iter=7, eps_iter=0.1, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
criterion = nn.CrossEntropyLoss()

def test():
    print('\n[ Test Start ]')
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        benign_loss += loss.item()

        _, predicted = outputs.max(1)
        benign_correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current benign test loss:', loss.item())

        adv = adversary.perturb(inputs, targets)
        adv_outputs = net(adv)
        loss = criterion(adv_outputs, targets)
        adv_loss += loss.item()

        _, predicted = adv_outputs.max(1)
        adv_correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print('Current adversarial test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current adversarial test loss:', loss.item())

    print('\nTotal benign test accuracy:', 100. * benign_correct / total)
    print('Total adversarial test accuracy:', 100. * adv_correct / total)
    print('Total benign test loss:', benign_loss)
    print('Total adversarial test loss:', adv_loss)

if __name__ == '__main__':
    test()
