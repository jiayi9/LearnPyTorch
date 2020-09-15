
import torch
import time


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # if no divice specified, use the device of the net
        device = list(net.parameters())[0].device 
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # eval mode, this will shutdown the dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # change back to train mode
            else: # defined model, will not be used after 3.13, no  consi9dering GPUI
                if('is_training' in net.__code__.co_varnames): # if there is is_training
                    # set is_training into False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item() 
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item() 
            n += y.shape[0]
    return acc_sum / n

def train_3c(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    EPOCH, train_loss, train_accuracy, val_accuracy = [], [], [], []
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for X, y in train_iter:
        #for index, (X, y) in enumerate(train_iter):
            if index % 10 == 0:
              print(index)
            #X = X[:,0,:,:].unsqueeze(1)
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
        
        EPOCH.append(epoch)
        train_loss.append(train_l_sum / batch_count)
        train_accuracy.append(train_acc_sum / n)
        val_accuracy.append(val_acc)
    return {'EPOCH': EPOCH, 'train_loss':train_loss, 'train_accuracy':train_accuracy, 'val_accuracy':val_accuracy}


def train_1c(net, train_iter, val_iter, batch_size, optimizer, device, num_epochs):
    EPOCH, train_loss, train_accuracy, val_accuracy = [], [], [], []
    net = net.to(device)
    print("training on ", device)
    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        #for X, y in train_iter:
        for index, (X, y) in enumerate(train_iter):
            #if index % 10 == 0:
            #  print(index)
            #X = X[:,0,:,:].unsqueeze(1)
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        val_acc = evaluate_accuracy(val_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, val acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, val_acc, time.time() - start))
        EPOCH.append(epoch)
        train_loss.append(train_l_sum / batch_count)
        train_accuracy.append(train_acc_sum / n)
        val_accuracy.append(val_acc)
    return {'EPOCH': EPOCH, 'train_loss':train_loss, 'train_accuracy':train_accuracy, 'val_accuracy':val_accuracy}

        
