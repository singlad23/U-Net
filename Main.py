import os
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
# from torchsummary import summary

import Model
from Dataset import UCF101
from Utils import build_paths, print_time, set_seed


print_time('START TIME')

#### Paths #############################################################################################################

class_idxs, train_split, test_split, frames_root, pretrained_path = build_paths()

#### Params ############################################################################################################

print('\n==> Initializing Hyperparameters...\n')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0                                                           # best test accuracy
start_epoch = 0                                                        # start from epoch 0 or last checkpoint epoch
num_epochs = 200
initial_lr = .0001
batch_size = 30
num_workers = 2
num_classes = 101
seed = 0
clip_len = 16
model_summary = False
resume = False
pretrain = True
print_batch = False
nTest = 1

accuracyTrain = []
lossTrain = []

accuracyTest = []
lossTest = []

set_seed(seed=seed)

print('GPU Support:', 'Yes' if device != 'cpu' else 'No')
print('Starting Epoch:', start_epoch)
print('Total Epochs:', num_epochs)
print('Batch Size:', batch_size)
print('Clip Length: ', clip_len)
print('Initial Learning Rate: %g' % initial_lr)
print('Random Seed:', seed)

### Data ###############################################################################################################

print('\n==> Preparing Data...\n')

trainset = UCF101(class_idxs=class_idxs, split=train_split, frames_root=frames_root, clip_len=clip_len, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

testset = UCF101(class_idxs=class_idxs, split=test_split, frames_root=frames_root, clip_len=clip_len, train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

print('Number of Classes: %d' % num_classes)
print('Number of Training Videos: %d' % len(trainset))
print('Number of Testing Videos: %d' % len(testset))


### Model ##############################################################################################################

print('\n==> Building Model...\n')

model = Model.C3D(num_classes=num_classes, pretrained_path=pretrained_path, pretrained=pretrain, resume=resume)
model = model.to(device)

if model_summary:
    summary(model, input_size=(3, clip_len, 112, 112))


### Optimizer, Loss, initial_lr Scheduler ##############################################################################

train_params = [{'params': Model.get_1x_lr_params(model), 'initial_lr': initial_lr},
                {'params': Model.get_10x_lr_params(model), 'initial_lr': initial_lr * 10}]

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(train_params, lr=initial_lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200], gamma=0.8)
criterion.to(device)

print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))


### Training ###########################################################################################################

def train(epoch):
    # print('\n==> Training model...\n')
    start = timer()
    # scheduler.step()
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        probs = nn.Softmax(dim=1)(outputs)
        predicted = torch.max(probs, 1)[1]
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if print_batch:
            print('Epoch: %d | Batch: %d/%d | Running Loss: %.3f | Running Acc: %.2f%% (%d/%d) [Train]'
                % (epoch+1, batch_idx+1, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    end = timer()
    optim_dict = optimizer.state_dict()
    current_lr = optim_dict['param_groups'][0]['lr']

    print('Epoch %d | Loss: %.3f | Acc: %.2f%% | Current lr: %f | Time: %.2f min [Train]'
                % (epoch+1, train_loss/len(trainloader), 100.*correct/total, current_lr, (end - start)/60))
    accuracyTrain.append(100. * correct / total)
    lossTrain.append(train_loss / len(trainloader))


### Testing ############################################################################################################

def test(epoch):
    # print('\n==> Testing model...\n')
    start = timer()
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if print_batch:
                print('Epoch: %d | Batch: %d/%d | Running Loss: %.3f | Running Acc: %.2f%% (%d/%d) [Test]'
                    % (epoch+1, batch_idx+1, len(testloader), test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    end = timer()
    optim_dict = optimizer.state_dict()
    current_lr = optim_dict['param_groups'][0]['lr']

    print('Epoch %d | Loss: %.3f | Acc: %.2f%% | Current lr: %f | Time: %.2f min [Test]'
          % (epoch+1, test_loss/len(testloader), 100.*correct/total, current_lr, (end - start)/60))
    accuracyTest.append(100. * correct / total)
    lossTest.append(test_loss / len(testloader))

    # Save checkpoint
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving Checkpoint..')
        state = {'model_state': model.state_dict(), 'optim_state': optimizer.state_dict(), 'acc': acc, 'epoch': epoch}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
        
        
# RUNNNNNN

lr = initial_lr


for epoch in range(start_epoch, start_epoch+200):
    if epoch == start_epoch:
        print('\n==> Training model...\n')

    train(epoch)

    if (epoch + 1) % nTest == 0:
        test(epoch)
    
    # Update the learning rate according to the scheduler
    scheduler.step()
    if lr != scheduler.get_lr()[0]:
        lr = scheduler.get_lr()[0]
        print('\nlr = %.g\n' % lr)

fig1 = plt.figure(0)
plt.plot(accuracyTrain)
plt.plot(accuracyTest)
plt.title('Accuracy UCF101 Pytorch')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig1.savefig('accuracyPlot')

fig2 = plt.figure(1)
plt.plot(lossTrain)
plt.plot(lossTest)
plt.title('Loss UCF101 Pytorch')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
fig2.savefig('lossPlot')

