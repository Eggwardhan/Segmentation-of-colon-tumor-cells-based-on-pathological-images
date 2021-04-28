# -*- coding: utf-8 -*-

import torch
import torchvision.models as models
from torchsummary import summary
import model.model as model
device =torch.device("cuda:1")
#device=torch.device("cpu")
net = model.choose_net("nested_unet")
net = net(in_channel=3,out_channel=1)
load = "./check_point/nested_unetbatch4scale512.0epoch30.pth"
net.load_state_dict(
    #torch.load(load,map_location=device)
    torch.load(load)

)
#net.to(device)
indata=torch.FloatTensor(3,512,512)

#summary(net,(3,512,512))
para={}
for name,parameters in net.named_parameters():
    print(name,":",parameters.size())
    para[name]=parameters.detach().numpy()
print(para)

