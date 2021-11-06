import time

import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision import transforms

from Data import *
from Model import VGG
from Loss import Loss
from Utils import *

print_time('START TIME')

# To print out all of the hyper parameters for better organization

#### Initalize Parameters ##############################################################################################

print('==> Initializing Parameters...\n')

data_path = './data'

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # if a GPU is available use it otherwise use the CPU
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_epochs = 200
batch_size = 128
initial_lr = .01
seed = 0
resume = False
num_classes = 10
model_name = 'VGG16'

set_seed(seed=seed)

print('GPU Support:', 'Yes' if device != 'cpu' else 'No')
print('Starting Epoch:', start_epoch)
print('Total Epochs:', num_epochs)
print('Batch Size:', batch_size)
print('Initial Learning Rate: %g' % initial_lr)
print('Random Seed:', seed)
print('Resuming Training:', 'Yes' if resume else 'No')

#### Data ##############################################################################################################

print('\n==> Preparing Data...\n')

mean, std_dev = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)  # Pre-calculated for every dataset

# Train Data
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                      # Transform my traniing data, takes a random crop of a square of 32 pixels out
                                      # of the image each time it wants to train with it (to prevent overfitting by
                                      # considering different parts of the image each epoch), horizontal flip does a
                                      # similar thing by preventing overfitting, padding = 4 just adds four
                                      # additional pixels around the image to prevent an exception from happening (
                                      # due to the random cropping)
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, std_dev)])  # .ToTensor is something GPUs can act on
# .Normalize gives it the mean and standard deviation above for the four channels

trainset = CIFAR10(root=data_path, train=True, download=True, transform=transform_train)  # Instantiate the dataset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                          num_workers=0)  # changed from 2 to 0
train_size = trainset.__len__()

# Test Data
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std_dev)])
testset = CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)  # changed from 2 to 0
test_size = testset.__len__()

print('Number of Train Samples:', train_size)
print('Number of Test Samples:', test_size)
print('Number of Classes:', num_classes)

#### Model #############################################################################################################

print('\n==> Building model...\n')

net = VGG(model_name)  # this builds the model
net = net.to(device)  # selects whether to run the model on the GPU or CPU (whichever one is available)

#### Resume from Checkpoint ############################################################################################

if resume:
    print('==> Resuming from checkpoint...\n')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')  # loads the model weights, NAME OF THE WEIGHTS LOADED MUST BE THE
    # SAME AS YOURS OTHERWISE THIS WILL NOT THROW AN ERROR BUT IT WILL ALSO NOT WORK
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    print('Best Accuracy: %g%%' % best_acc)
    print('Resuming at Epoch: %d\n' % start_epoch)

### Optimizer & Loss ###################################################################################################

criterion = Loss()  # Which loss function will I use? I will use my own!
optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4)  # uses SGD optimizer
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[42], gamma=0.1)  # changes the learning rate as you


# go along, every 42 epochs you increase your learning rate by 0.1


#### Training ##########################################################################################################

def train(epoch):
    since = time.time()  # measure the time it takes to train
    net.train()  # weights will update because program knows it's training
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets, img_ids) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)     # send the inputs to the GPU/CPU
        optimizer.zero_grad()  # zero out the gradient (to prevent miscalculation) from the previous run
        outputs = net(inputs)  # get the outputs from the model (net)

        loss = criterion(outputs, targets)  # loss function (ours)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()   # add the loss to the list of loss
        _, predicted = outputs.max(dim=1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    time_elapsed = time.time() - since
    hrs, _min, sec = time_elapsed // 3600, time_elapsed // 60, time_elapsed % 60

    print('Epoch: %d | Time: %02d:%02d:%02d | Loss: %.3f | Acc: %.2f%% [Train]'     # print out everything
          % (epoch, hrs, _min, sec, train_loss / len(trainloader), 100. * correct / total))


#### Testing ###########################################################################################################

def test(epoch):
    since = time.time()
    global best_acc     # if my model improves, then I want to update my accuracy based on what it was before
    # (save the model weights everytime they give the best accuracy)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():   # I am not upgrading the weights in my model because this is test, so don't do that here
        for batch_idx, (inputs, targets, img_ids) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(dim=1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    time_elapsed = time.time() - since
    hrs, _min, sec = time_elapsed // 3600, time_elapsed // 60, time_elapsed % 60

    print('Epoch: %d | Time: %02d:%02d:%02d | Loss: %.3f | Acc: %.2f%% [Test]'
          % (epoch, hrs, _min, sec, test_loss / len(testloader), 100. * correct / total))

    # Save checkpoint
    acc = 100. * correct / total
    if acc > best_acc:  # if the accuracy is best than the supposed highest global accuracy
        print('Saving Checkpoint..')
        state = {'net': net.state_dict(), 'acc': acc, 'epoch': epoch}  # save the new model weights, accuracy, and epoch
        if not os.path.isdir('checkpoint'):     # if this isn't already a directory
            os.mkdir('checkpoint')      # then make it one
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


#### Run ###############################################################################################################

lr = initial_lr

for epoch in range(start_epoch, start_epoch + 200):
    if epoch == start_epoch:
        print('==> Training model...\n')
        print('lr = %g\n' % lr)
    train(epoch)  # train for each epoch
    test(epoch)  # test for each epoch

    # Update the learning rate according to the scheduler
    scheduler.step()
    if lr != scheduler.get_lr()[0]:
        lr = scheduler.get_lr()[0]
        print('\nlr = %.g\n' % lr)

plt.figure(0)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
