import syft as sy

class Deep2DNet(sy.Module):
    """ Three sets of Conv2d layers with Max2DPooling.
    
    Arguments:
        torch_ref (object): Needs to be given a reference to torch for initialization.
    """
    def __init__(self, torch_ref):
        super(Deep2DNet, self).__init__(torch_ref=torch_ref)
        
        self.conv1_1 = self.torch_ref.nn.Conv2d(1, 32, 7)
        self.conv1_2 = self.torch_ref.nn.Conv2d(32, 64, 7)
        
        self.conv2_1 = self.torch_ref.nn.Conv2d(64, 128, 5)
        self.conv2_2 = self.torch_ref.nn.Conv2d(128, 256, 5)
        
        self.conv3_1 = self.torch_ref.nn.Conv2d(256, 256, 3)
        self.conv3_2 = self.torch_ref.nn.Conv2d(256, 256, 3)
        
        # Calculate convs output size in initialization
        x = self.torch_ref.randn(64, 64).view(-1, 1, 64, 64)
        self._to_linear = None
        self.convs(x)
        
        self.fc1 = self.torch_ref.nn.Linear(self._to_linear, 512)
        self.fc2 = self.torch_ref.nn.Linear(512, 6)
        
    def convs(self, x):
         # First Block
        x = self.torch_ref.nn.functional.relu(self.conv1_1(x))
        x = self.torch_ref.nn.functional.max_pool2d(self.torch_ref.nn.functional.relu(self.conv1_2(x)), 2)
        # Second Block
        x = self.torch_ref.nn.functional.relu(self.conv2_1(x))
        x = self.torch_ref.nn.functional.max_pool2d(self.torch_ref.nn.functional.relu(self.conv2_2(x)), 2)
        # Thrid Block
        x = self.torch_ref.nn.functional.relu(self.conv3_1(x))
        x = self.torch_ref.nn.functional.max_pool2d(self.torch_ref.nn.functional.relu(self.conv3_2(x)), 2)
        
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
        
    def forward(self, x):
        x = self.convs(x)
        # Fully connected Block
        x = self.torch_ref.flatten(x, start_dim=1)
        x = self.torch_ref.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return self.torch_ref.nn.functional.log_softmax(x, dim=1)
        

class Shallow3DNet(sy.Module):
    """ Three Conv3D layers with MaxPool3D inbetween.
    
    Arguments:
        torch_ref (object): Needs to be given a reference to torch for initialization.
    """
    def __init__(self, torch_ref):
        super(Shallow3DNet, self).__init__(torch_ref=torch_ref)
        self.conv1 = self.torch_ref.nn.Conv3d(1, 32, 5)
        self.conv2 = self.torch_ref.nn.Conv3d(32, 64, 5)
        self.conv3 = self.torch_ref.nn.Conv3d(64, 128, 5)
        
        self.pool = self.torch_ref.nn.MaxPool3d(3)
        
        x = self.torch_ref.randn(96,96,96).view(-1, 1, 96, 96, 96)
        self._to_linear = None
        self.convs(x)

        self.fc1 = self.torch_ref.nn.Linear(self._to_linear, 512)
        self.fc2 = self.torch_ref.nn.Linear(512, 2)
        
    def convs(self, x):
        x = self.pool(self.torch_ref.nn.functional.relu(self.conv1(x)))
        x = self.pool(self.torch_ref.nn.functional.relu(self.conv2(x)))
        x = self.pool(self.torch_ref.nn.functional.relu(self.conv3(x)))

        # We need to know the input dims for the linear layer and flatten
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear) # Flatten for fc layer
        x = self.torch_ref.nn.functional.relu(self.fc1(x))
        x = self.fc2(x) # No activation bcs this is the output layer
        return self.torch_ref.nn.functional.log_softmax(x, dim=1)
    