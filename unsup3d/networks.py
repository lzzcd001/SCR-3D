import torch
import torch.nn as nn
import torchvision


EPS = 1e-7


class SmallEncoder(nn.Module):
    def __init__(self, cin, zdim=128, nf=64, **kwargs):
        super(SmallEncoder, self).__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, zdim, kernel_size=1, stride=1, padding=0, bias=False)]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input)


class BigEncoder(nn.Module):
    def __init__(self, cin, zdim=128, nf=64, **kwargs):
        super(BigEncoder, self).__init__()
        ## downsampling
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True),
            nn.Conv2d(zdim, zdim, kernel_size=1, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
        ]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input)



class SmallDecoder(nn.Module):
    def __init__(self, cout, zdim=128, nf=64, activation=nn.Tanh, **kwargs):
        super(SmallDecoder, self).__init__()
        network = [
            nn.Conv2d(zdim, zdim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.Conv2d(zdim, cout, kernel_size=1, stride=1, padding=0, bias=False),
        ]
        self.network = nn.Sequential(*network)
        if activation is not None:
            network += [activation()]

    def forward(self, input):
        return self.network(input).reshape(input.size(0),-1)


class BigDecoder(nn.Module):
    def __init__(self, cout, zdim=128, nf=64, activation=nn.Tanh, **kwargs):
        super(BigDecoder, self).__init__()
        network = [
            nn.ConvTranspose2d(zdim, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, cout, kernel_size=5, stride=1, padding=2, bias=False)]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input)

class ResMLPBlock(nn.Module):
    def __init__(self, chidden):
        super(ResMLPBlock, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(chidden, chidden, 1, 1, 0),
            # nn.BatchNorm2d(chidden),
            nn.GroupNorm(32, chidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(chidden, chidden, 1, 1, 0),
        )
        self.act = nn.ReLU(inplace=True)
        # self.bn = nn.BatchNorm2d(chidden)
        self.bn = nn.GroupNorm(32, chidden)
    
    def forward(self, x):
        return self.act(self.bn(x + self.network(x)))

class FactorMLP(nn.Module):
    def __init__(self, cin, cemb):
        super(FactorMLP, self).__init__()
        network = [
            nn.Conv2d(cin, cemb, 1, 1, 0),
            # ResMLPBlock(cemb),
            # ResMLPBlock(cemb),
            # ResMLPBlock(cemb),
        ]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input)


# class Encoder(nn.Module):
#     def __init__(self, cin, cout, nf=64, activation=nn.Tanh):
#         super(Encoder, self).__init__()
#         network = [
#             nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
#             nn.ReLU(inplace=True),
#             nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
#             nn.ReLU(inplace=True),
#             nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
#             nn.ReLU(inplace=True),
#             nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
#             nn.ReLU(inplace=True),
#             nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
#             nn.ReLU(inplace=True),
#             nn.Conv2d(nf*8, cout, kernel_size=1, stride=1, padding=0, bias=False)]
#         if activation is not None:
#             network += [activation()]
#         self.network = nn.Sequential(*network)

#     def forward(self, input):
#         return self.network(input).reshape(input.size(0),-1)


# class EDDeconv(nn.Module):
#     def __init__(self, cin, cout, zdim=128, nf=64, activation=nn.Tanh):
#         super(EDDeconv, self).__init__()
#         ## downsampling
#         network = [
#             nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
#             nn.GroupNorm(16, nf),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
#             nn.GroupNorm(16*2, nf*2),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
#             nn.GroupNorm(16*4, nf*4),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
#             nn.ReLU(inplace=True)]
#         ## upsampling
#         network += [
#             nn.ConvTranspose2d(zdim, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 1x1 -> 4x4
#             nn.ReLU(inplace=True),
#             nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
#             nn.GroupNorm(16*4, nf*4),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.GroupNorm(16*4, nf*4),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 16x16
#             nn.GroupNorm(16*2, nf*2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.GroupNorm(16*2, nf*2),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
#             nn.GroupNorm(16, nf),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.GroupNorm(16, nf),
#             nn.ReLU(inplace=True),
#             nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64
#             nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.GroupNorm(16, nf),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=False),
#             nn.GroupNorm(16, nf),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(nf, cout, kernel_size=5, stride=1, padding=2, bias=False)]
#         if activation is not None:
#             network += [activation()]
#         self.network = nn.Sequential(*network)

#     def forward(self, input):
#         return self.network(input)


class ConfNet(nn.Module):
    def __init__(self, cin, cout, zdim=128, nf=64):
        super(ConfNet, self).__init__()
        ## downsampling
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True)]
        ## upsampling
        network += [
            nn.ConvTranspose2d(zdim, nf*8, kernel_size=4, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True)]
        self.network = nn.Sequential(*network)

        out_net1 = [
            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 64x64
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, 2, kernel_size=5, stride=1, padding=2, bias=False),  # 64x64
            nn.Softplus()]
        self.out_net1 = nn.Sequential(*out_net1)

        out_net2 = [nn.Conv2d(nf*2, 2, kernel_size=3, stride=1, padding=1, bias=False),  # 16x16
                    nn.Softplus()]
        self.out_net2 = nn.Sequential(*out_net2)

    def forward(self, input):
        out = self.network(input)
        return self.out_net1(out), self.out_net2(out)


class PerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(PerceptualLoss, self).__init__()
        mean_rgb = torch.FloatTensor([0.485, 0.456, 0.406])
        std_rgb = torch.FloatTensor([0.229, 0.224, 0.225])
        self.register_buffer('mean_rgb', mean_rgb)
        self.register_buffer('std_rgb', std_rgb)

        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def normalize(self, x):
        out = x/2 + 0.5
        out = (out - self.mean_rgb.view(1,3,1,1)) / self.std_rgb.view(1,3,1,1)
        return out

    def __call__(self, im1, im2, mask=None, conf_sigma=None):
        im = torch.cat([im1,im2], 0)
        im = self.normalize(im)  # normalize input

        ## compute features
        feats = []
        f = self.slice1(im)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice2(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice3(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice4(f)
        feats += [torch.chunk(f, 2, dim=0)]

        losses = []
        for f1, f2 in feats[2:3]:  # use relu3_3 features only
            loss = (f1-f2)**2
            if conf_sigma is not None:
                loss = loss / (2*conf_sigma**2 +EPS) + (conf_sigma +EPS).log()
            if mask is not None:
                b, c, h, w = loss.shape
                _, _, hm, wm = mask.shape
                sh, sw = hm//h, wm//w
                mask0 = nn.functional.avg_pool2d(mask, kernel_size=(sh,sw), stride=(sh,sw)).expand_as(loss)
                loss = (loss * mask0).sum() / mask0.sum()
            else:
                loss = loss.mean()
            losses += [loss]
        return sum(losses)
