def image_classify(fname):
    import torch
    from torch import nn

    classifier = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),  
                                nn.ReLU(),
                                nn.BatchNorm2d(num_features=32),
                                nn.MaxPool2d(kernel_size=2),
                                nn.Conv2d(32, 32, 3),
                                nn.ReLU(),
                                nn.BatchNorm2d(32),
                                nn.MaxPool2d(2),
                                nn.Flatten(),
                                nn.Linear(in_features=14*14*32, out_features=128),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(128, 128),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(128, 1),
                                nn.Sigmoid())
    
    device = torch.device('cpu') 
    classifier.load_state_dict(torch.load('classifier.pth', map_location = device))

    from PIL import Image

    test_image = Image.open(fname)

    import numpy as np
    test_image = test_image.convert("RGB")
    test_image = test_image.resize((64,64))
    test_image = np.array(test_image.getdata()).reshape(*test_image.size, 3)
    test_image = test_image / 255
    test_image = test_image.transpose(2, 0, 1)
    test_image = torch.tensor(test_image, dtype=torch.float).view(-1, *test_image.shape)

    classifier.eval()
    test_image = test_image.to(device)

    output = classifier.forward(test_image)

    if output > 0.5:
        output = 1
    else:
        output = 0

    print('Forecast: ', output)

    idx_to_class = {1: 'You have sent a Cat 🐱', 0: 'You have sent a Dog 🐶'}

    return idx_to_class[output]
