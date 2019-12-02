import torch
from model import Bowman
from data import SNLI

def bowman_train(model, dataset, criterion, optimizer, epoch_num=5):
    snli = dataset
    record = open("result.txt", "wb", buffering=0)
    model.train()
    for epoch in range(epoch_num):

        running_loss = 0.0
        epoch_loss = 0.0
        i = 0
        for batch in snli.train_iter:
            i += 1
            #get data
        
            premise, _ = batch.premise
            hypothesis, _ = batch.hypothesis
            label = batch.label

            # zeros the paramster gradients
            optimizer.zero_grad()       # 

            # forward + backward + optimize
            output = model(premise, hypothesis)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step() 

            # print statistics
            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % 1000 == 999:
                print('[%d, %5d] loss: %.3f' % (epoch, i + 1, running_loss / 1000))
                running_loss = 0.0
        print('epoch %d loss: %.3f\n' % (epoch, epoch_loss / (i + 1)))
        record.write(b'%f\n' % (epoch_loss / (i + 1)))
        torch.save(model, './model/bowman_%d.pth'% (epoch))

    