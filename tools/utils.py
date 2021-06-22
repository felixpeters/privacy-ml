import time
import numpy as np

def train(batch_size, epochs, delta, model, 
          torch_ref, optim, loss_function, 
          train_data, train_labels, test_data, 
          test_labels, input_shape, device, privacy_engine=None):
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
        privacy_engine (object): An instance of the Differential Privacy Engine from Opacus or a Pointer to a remote engine.
    """
    # Variables to track
    losses = [] # Training losses per batch per epoch
    test_accs = []
    test_losses = [] # Test losses per epoch
    epsilons = [] 
    alphas = []
    epoch_times = [] # Training times for each epoch
    best_acc_loss = (0, 0)
    best_model = None
    
    # Divide dataset into batches (sadly remote DataLoaders aren't yet a thing in pysyft)
    length = len(train_data)
    
    if length % batch_size != 0:
        cut_data = train_data[:length - length % batch_size]
        cut_labels = train_labels[:length - length % batch_size]
        
    shape = [-1, batch_size]
    shape.extend(input_shape)
    
    batch_data = cut_data.view(shape)
    batch_labels = cut_labels.view(-1, batch_size)
    
    # Prepare indices for randomization of order for each epoch
    indices = np.arange(int(length / batch_size))
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = []
        
        model.train()
        
        np.random.shuffle(indices)
        
        print(f'###### Epoch {epoch + 1} ######')
        for i in indices:
            optim.zero_grad()
            
            output = model(batch_data[int(i)].to(device))
            
            loss = loss_function(output, batch_labels[int(i)].to(device))
            loss_item = loss.item()
            
            if model.is_local:
                loss_value = loss_item
            else:
                loss_value = loss_item.get(reason="To evaluate training progress", request_block=True, timeout_secs=5)
            #print(f'Training Loss: {loss_value}')
            epoch_loss.append(loss_value)
        
            loss.backward()
            optim.step()
        
        # Checking our privacy budget
        if privacy_engine is not None:
            epsilon_tuple = privacy_engine.get_privacy_spent(delta)
            if model.is_local:
                epsilon = epsilon_tuple[0]
                best_alpha = epsilon_tuple[1]
            else:
                epsilon_ptr = epsilon_tuple[0].resolve_pointer_type()
                best_alpha_ptr = epsilon_tuple[1].resolve_pointer_type()

                epsilon = epsilon_ptr.get(
                    reason="So we dont go over it",
                    request_block=True,
                    timeout_secs=5
                )
                best_alpha = best_alpha_ptr.get(
                    reason="So we dont go over it",
                    request_block=True,
                    timeout_secs=5
                )
    
            if epsilon is None:
                epsilon = float("-inf")
            if best_alpha is None:
                best_alpha = float("-inf")
            print(
                f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha}"
            )
            epsilons.append(epsilon)
            alphas.append(best_alpha)
    
        test_acc, test_loss = test(model, loss_function, torch_ref, test_data, test_labels, device)
        print(f'Test Accuracy: {test_acc} ---- Test Loss: {test_loss}')
        
        epoch_end = time.time()
        print(f"Epoch time: {float(epoch_end - epoch_start)} seconds")
        
        losses.append(epoch_loss)
        epoch_times.append(float(epoch_end - epoch_start))
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        
    return losses, test_accs, test_losses, epsilons, alphas, epoch_times

            
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
        loss = test_loss.item()
    else:
        acc = acc_ptr.get(reason="To evaluate training progress", request_block=True, timeout_secs=5)
        loss = test_loss.item().get(reason="To evaluate training progress", request_block=True, timeout_secs=5)

    return acc, loss