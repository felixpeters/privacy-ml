def train(batch_size, epochs, model, 
          torch_ref, optim, loss_function, 
          train_data, train_labels, test_data, 
          test_labels, input_shape, device):
    """ A generic training function without the use of data loaders.
    
    Arguments:
        batch_size (int): Size of the mini batches. Length of data should be close to a multiple of this since the rest is
                            cut from the training data!
        epochs (int): How often should the training process iterate over whole dataset.
        model (object): Remote or local model.
        torch_ref (object): Remote or local torch reference.
        optim (object): Remote or local optimizer.
        loss_function (object): Remote or local loss function.
        train_data (torch.Tensor): A Tensor or TensorPointer conatining the training data.
        train_labels (torch.Tensor): A Tensor or TensorPointer containing the training labels.
        test_data (torch.Tensor): A Tensor or TensorPointer conatining the test data.
        test_labels (torch.Tensor): A Tensor or TensorPointer containing the test labels.
        input_shape (list): A list specifying the shape of one data point (result of train_data[0].shape).
        device (torch.device): The device to train on.
    """
    length = len(train_data)
    
    if length % batch_size != 0:
        cut_data = train_data[:length - length % batch_size]
        cut_labels = train_labels[:length - length % batch_size]
        
    shape = [-1, batch_size]
    shape.extend(input_shape)
    
    batch_data = cut_data.view(shape)
    batch_labels = cut_labels.view(-1, batch_size)
    
    for epoch in range(epochs):
        model.train()
        
        print(f'###### Epoch {epoch + 1} ######')
        for i in range(int(length / batch_size)):
            optim.zero_grad()
            
            output = model(batch_data[i].to(device))
            
            loss = loss_function(output, batch_labels[i].to(device))
            loss_item = loss.item()
            loss_value = loss_item.get_copy(reason="To evaluate training progress", request_block=True, timeout_secs=5)
            print(f'Training Loss: {loss_value}')
        
            loss.backward()
            optim.step()
        
        test(model, loss_function, torch_ref, test_data, test_labels, device)
                   
            
def test(model, loss_function, remote_torch, data, labels, device):
    model.eval()
    
    data = data.to(device)
    labels = labels.to(device)
    length = len(data)
    
    with remote_torch.no_grad():
        output = model(data)
        test_loss = loss_function(output, labels)
        prediction = output.argmax(dim=1)
        total = prediction.eq(labels).sum().item()
        
    acc_ptr = total / length
    acc = acc_ptr.get(request_block=True, reason='Gimme!')
    loss = test_loss.get(request_block=True, reason='Gimme!')
    
    print(f'Test Accuracy: {acc} --- Test Loss: {loss}')