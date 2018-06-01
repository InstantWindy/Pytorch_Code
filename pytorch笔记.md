#1. pytorch中的最近采样和双线性采样
    F.upsample_nearest:要求输出和输入之间是整数倍关系
    F.upsample_billinar:不要求
#2. 模型参数更新
    res50=resnet50(pretrained=False).cuda()
    net=ICNet(num_classes,True).cuda()
    feats=res50(im)
    out1,out2,out3=net(im,feats)
    optimizer=torch.optim.SGD(net.parameters(),lr=1e-2,momentum=0.9,weight_decay=1e-4)
	

 - 只对net中的参数进行了优化更新，res50中的参数没有进行更新
 - 如果想对res50和net的参数都进行跟新，需要把res50写入到net中
 
#3. 均值和方差
    沿着H,W求均值和方差
    var=img.view(N, C, -1).var(dim=2)
    var=var.view(N, C, 1, 1)
