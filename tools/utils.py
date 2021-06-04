def train(batch_size, epochs, model, 
          torch_ref, optim, loss_function, 
          train_data, train_labels, test_data, 
          test_labels, input_shape, device):
    """ A generic training function without the use of data loaders and test cycle after every epoch.
    
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
            
            if model.is_local:
                loss_value = loss_item
            else:
                loss_value = loss_item.get_copy(reason="To evaluate training progress", request_block=True, timeout_secs=5)
            print(f'Training Loss: {loss_value}')
        
            loss.backward()
            optim.step()
        
        test(model, loss_function, torch_ref, test_data, test_labels, device)
                   
            
def test(model, loss_function, torch_ref, data, labels, device):
    """ A generic training function without batching.
    
     Arguments:
        model (object): Remote or local model.
        torch_ref (object): Remote or local torch reference.
        loss_function (object): Remote or local loss function.
        data (torch.Tensor): A Tensor or TensorPointer conatining the test data.
        labels (torch.Tensor): A Tensor or TensorPointer containing the test labels.
        device (torch.device): The device to train on.
    """
    model.eval()
    
    data = data.to(device)
    labels = labels.to(device)
    length = len(data)
    
    with torch_ref.no_grad():
        output = model(data)
        test_loss = loss_function(output, labels)
        prediction = output.argmax(dim=1)
        total = prediction.eq(labels).sum().item()
        
    acc_ptr = total / length
    if model.is_local:
        acc = acc_ptr
        loss = test_loss
    else:
        acc = acc_ptr.get(reason="To evaluate training progress", request_block=True, timeout_secs=5)
        loss = test_loss.get(reason="To evaluate training progress", request_block=True, timeout_secs=5)
    
    print(f'Test Accuracy: {acc} --- Test Loss: {loss}')