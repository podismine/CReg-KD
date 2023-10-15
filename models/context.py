import torch
import torch.nn as nn
import torch.nn.functional as F

class Refine(nn.Module):

    def __init__(self,channels,config):
        super(Refine, self).__init__()
        self.ab = config.ab
        self.conv1 = nn.Sequential(
            nn.Conv3d(channels[0], channels[0], kernel_size=3, stride=1, padding=1,
                          bias=False, groups=channels[0]),
                nn.Conv3d(channels[0], channels[0], kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(channels[0]),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels[1], channels[1], kernel_size=3, stride=1, padding=1,
                          bias=False, groups=channels[1]),
                nn.Conv3d(channels[1], channels[1], kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(channels[1]),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(channels[2], channels[2], kernel_size=3, stride=1, padding=1,
                          bias=False, groups=channels[2]),
                nn.Conv3d(channels[2], channels[2], kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(channels[2]),
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(channels[3], channels[3], kernel_size=3, stride=1, padding=1,
                          bias=False, groups=channels[3]),
                nn.Conv3d(channels[3], channels[3], kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(channels[3]),
        )
    def forward(self, x_t, x_s,weight):

        fea_t = x_t['fea']
        loss = 0.
        fea_s = x_s['fea']
        t1 = self.conv1(fea_s[0])
        t2 = self.conv2(fea_s[1])
        t3 = self.conv3(fea_s[2])
        t4 = self.conv4(fea_s[3])
        loss += 1 * sum([self.at_loss(f_s, f_t.detach(), weight) for f_s, f_t in zip([t1,t2,t3,t4], fea_t)])
        
        return loss

    def at_loss(self, f_s, f_t, weight):
        out = ((self.at(f_s) - self.at(f_t)).pow(2)* weight[:,None]).mean() 
        #out = ((self.at(f_s) - self.at(f_t)).pow(2)).mean()
        return out
    def at(self, f):
        if self.ab == 0:
            xx = f.pow(2).mean(1).view(f.size(0), -1)     
        elif self.ab == 1:
            xx = f.mean(1).pow(2).view(f.size(0), -1)   
        elif self.ab == 2:
            xx = f.mean(1).view(f.size(0), -1)    
        elif self.ab == 3:
            xx = f.view(f.size(0), -1)    
        return xx
        #return F.normalize(xx)
        #return F.normalize(f.pow(2).mean(1).view(f.size(0), -1))


import torch.nn as nn
import torch.nn.functional as F

class DistillKL(nn.Module):
    def __init__(self, args):
        super(DistillKL, self).__init__()
        self.T = args.temperature

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t.detach(), reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss

class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.p = 2
        self.kd = DistillKL(args)
        self.alpha = args.alpha
        self.beta = args.beta

    def forward(self, o_s, o_t, g_s, g_t):
        loss = self.alpha * self.kd(o_s, o_t)
        loss += self.beta * sum([self.at_loss(f_s, f_t.detach()) for f_s, f_t in zip(g_s, g_t)])

        return loss

    def at_loss(self, f_s, f_t):
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))

