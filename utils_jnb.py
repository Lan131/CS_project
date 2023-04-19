import torch 
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models, datasets
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchsummary import summary
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
import time
import torch.nn.functional as nnf
import torchattacks
import random

directory = "./src"



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def set_global_seed(seed=123):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    



class myData(Dataset):

    train_file = "training.csv"
    test_file = "testing.csv"

    def __init__(self, data_dir, transform=None, train=True):
        
        # The transform is goint to be used on image
        self.transform = transform
        
        if train:
            self.data_dir = os.path.join(data_dir,"train")
            data_file = self.train_file
        else:
            self.data_dir = os.path.join(data_dir,"test")
            data_file = self.test_file
        
        data_dircsv_file=os.path.join(self.data_dir,data_file)
        # Load the CSV file contians image info
        self.data_name= pd.read_csv(data_dircsv_file)
        
        # Number of images in dataset
        self.len=self.data_name.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        
        # Image file path
        img_name = os.path.join(self.data_dir,self.data_name.iloc[idx, 0])
        # Open image file
        image = Image.open(img_name)
        
        # The class label for the image
        y = self.data_name.iloc[idx, 1]

        target = {}
        target = {'none':0, 'building':1}

        y = target[y]
        
        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y

class splitData(Dataset):

    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, index):
        x, y = self.subset[index]
        return x, y
        
    def __len__(self):
        return len(self.subset)



def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


def makeTransform():
    transform_train = transforms.Compose([transforms.RandomResizedCrop(size=256, scale = (0.8,1.0)),
                                transforms.CenterCrop(size=224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])

    transform_test = transforms.Compose([transforms.Resize(size=256),
                                transforms.CenterCrop(size=224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])

    transform = {}
    transform = {'train': transform_train, 'test': transform_test}

    return transform

def getData():
    transform = makeTransform()

    train_valid = myData(data_dir=directory, transform=transform['train'], train=True)

    lengths = [int(np.floor(len(train_valid)*0.9)), int(np.ceil(len(train_valid)*0.1))]
    print(lengths )
    print(len(train_valid))
    train, val = random_split(train_valid, lengths)

    train_set = splitData(train)
    val_set = splitData(val)
    test_set = myData(data_dir=directory, transform=transform['test'], train=False)

    train_length = len(train_set)
    val_length = len(val_set)
    test_length = len(test_set)

    trainloader = DataLoader(dataset = train_set, batch_size = 32, shuffle=False, drop_last=True)
    valloader = DataLoader(dataset = val_set, batch_size = 4, shuffle=False, drop_last=True)
    testloader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, drop_last=False)

    return trainloader, valloader, testloader, train_length, val_length, test_length

def trainValid(model, lossDeterminer, optimizer, epochs=2):
    '''
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''
    trainloader, valloader, _, train_length, val_length, _ = getData()

    start = time.time()
    history = []
    bestAcc = 0.0

    for epoch in range(epochs):
        epochStart = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        
        # Set to training mode
        model.train()
        
        # Loss and Accuracy within the epoch
        trainLoss = 0.0
        trainAcc = 0.0
        
        validLoss = 0.0
        validAcc = 0.0
        
        for i, (inputs, labels) in enumerate(trainloader):

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Clean existing gradients
            optimizer.zero_grad()
            
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            
            # Compute loss
            loss = lossDeterminer(outputs, labels)
            
            # Backpropagate the gradients
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            
            # Compute the total loss for the batch and add it to trainLoss
            trainLoss += loss.item() * inputs.size(0)
            
            # Compute the accuracy
            _, predictions = torch.max(outputs.data, 1)
            corrCounts = predictions.eq(labels.data.view_as(predictions))
            
            # Convert corrCounts to float and then compute the mean
            acc = torch.mean(corrCounts.type(torch.FloatTensor))
            
            # Compute total accuracy in the whole batch and add to trainAcc
            trainAcc += acc.item() * inputs.size(0)
            
            print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

            
        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(valloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = lossDeterminer(outputs, labels)

                # Compute the total loss for the batch and add it to validLoss
                validLoss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                corrCounts = predictions.eq(labels.data.view_as(predictions))

                # Convert corrCounts to float and then compute the mean
                acc = torch.mean(corrCounts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to validAcc
                validAcc += acc.item() * inputs.size(0)

                print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
            
        # Find average training loss and training accuracy
        trainLossAvg = trainLoss/train_length
        trainAccAvg = trainAcc/train_length

        # Find average training loss and training accuracy
        validLossAvg = validLoss/val_length
        validAccAvg = validAcc/val_length

        history.append([trainLossAvg, validLossAvg, trainAccAvg, validAccAvg])
                
        epochEnd = time.time()
    
        print("Epoch : {:03d}, Training: Loss : {:.4f}, Accuracy: {:.4f}%".format(epoch, trainLossAvg, trainAccAvg*100))
        print("Validation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(validLossAvg, validAccAvg*100, epochEnd-epochStart))
        
        # Save if the model has best accuracy till now
        if validAccAvg > bestAcc:
            bestAcc = validAccAvg
            best_epoch = epoch
            torch.save(model, directory+'_model_'+str(epoch)+'.pt')
            print("model for epoch {} saved".format(epoch))
            
        print("Best accuracy achieved so far : {:.4f} on epoch {}".format(bestAcc, best_epoch))    
    return history

def computeTestSetAccuracy(model, loss_criterion, trojan=False):
    transform = makeTransform()
    test_set = myData(data_dir=directory, transform=transform['test'], train=False)
    testloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    
    correct = 0
    total = 0
    trojan_count = 0
    trojan_acc = 0
    
    with torch.no_grad():
        for j, (inputs, labels) in enumerate(testloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if trojan:
                image_name = testloader.dataset.data_name.iloc[j, 0]
                trojan_dict = get_trojan_dict()
                if trojan_dict[os.path.join('src', 'test', image_name)]:
                    trojan_count += 1
                    trojan_acc += (predicted == labels).sum().item()

    if trojan:
        if trojan_count > 0:
            avg_trojan_acc = trojan_acc / trojan_count
        else:
            avg_trojan_acc = 0

        if total - trojan_count > 0:
            avg_non_trojan_acc = (correct - trojan_acc) / (total - trojan_count)
        else:
            avg_non_trojan_acc = 0

        print("Trojan accuracy : " + str(avg_trojan_acc))
        print("Non-trojan accuracy : " + str(avg_non_trojan_acc))  
        
    # Calculate test accuracy and loss
    test_loss = 0.0
    test_acc = 0.0
    test_length = len(testloader.dataset)
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        loss = loss_criterion(outputs, labels)
        
        test_loss += loss.item()
        test_acc += (predicted == labels).sum().item()

    # Find average test loss and test accuracy
    avg_test_loss = test_loss/test_length
    avg_test_acc = test_acc/test_length

    print("Test accuracy : " + str(avg_test_acc))
    print("Test loss : " + str(avg_test_loss))

    if trojan:
        return avg_trojan_acc, avg_non_trojan_acc
    else:
        return avg_test_acc, avg_test_loss


def get_trojan_dict(path='./src/test/test_attack.csv'):
    """
    Returns a dictionary of image names with their corresponding trojan label (True/False).
    """
    trojan_dict = {}
    with open(path, 'r') as f:
        # Skip the header row
        next(f)
        for line in f:
            image_name, label = line.strip().split(',')
            if label.lower() == 'trojan':
                trojan_dict[os.path.join('src', 'test', image_name)] = True
            else:
                trojan_dict[os.path.join('src', 'test', image_name)] = bool(int(label))
    return trojan_dict


def plotCost(history):
    history = np.array(history)
    plt.plot(history[:,0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    #plt.ylim(0,1)
    plt.savefig(directory+'lossCurve.png')
    plt.show()

    plt.plot(history[:,2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0,1)
    plt.savefig(directory+'accuracyCurve.png')
    plt.show()
    
    
def predict_image(model_path,image_folder_path):
    try:
        saved_model = torch.load(model_path)
    except:
        saved_model = torch.load(model_path,map_location=torch.device('cpu'))
    transform_example = transforms.Compose([transforms.Resize(size=256),
                            transforms.CenterCrop(size=224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])])

    dataset = datasets.ImageFolder(image_folder_path, transform=transform_example)
     # Disable grad
    with torch.no_grad():
    
        # Retrieve item
        index = 0 #need a better way to reference examples
        item = dataset[index]
        image = item[0]
        #true_target = item[1]
        imshow(image)
        image=image.to(device)
        # Loading the saved model

    
        # Generate prediction
        prediction = saved_model(image.unsqueeze(0))
    
        # Predicted class value using argmax
        predicted_class = np.argmax(prediction.cpu())
        print("Class probability: " +str(torch.max(nnf.softmax(prediction, dim=1))))
    
        # Reshape image

        image=image.cpu()
        image=image.swapaxes(0,1)
        image=image.swapaxes(1,2)
    
        # Show result
        plt.imshow(image.cpu(), cmap='nipy_spectral')
        plt.title(f'Prediction: {predicted_class} - Actual target: 1')
        plt.show()

def adv_attack(type,model):
    transform_example= transforms.Compose([transforms.Resize(size=256),
                            transforms.CenterCrop(size=224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])])

    attackloader, _, _, attack_length, _, _ = getData()
    with torch.no_grad():
        for j, (inputs, labels) in enumerate(attackloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
        if type=='PDG':
            atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=4)
            return  atk(inputs, labels)

    
    

