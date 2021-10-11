
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models


class PerceptualLoss(nn.Module):
    def __init__(self, rank):
        super(PerceptualLoss, self).__init__()
        self.rank = rank
        self.vgg19 = torch_models.vgg19(pretrained=True)
        self.vgg19_relu_5_2 = nn.Sequential(*list(self.vgg19.features.children())[:-5]).eval()
        for p in self.vgg19_relu_5_2.parameters():
            p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input_, target):
        input_ = (input_ - self.mean) / self.std
        target = (target - self.mean) / self.std
        input_ = F.interpolate(input_, mode='bilinear', size=(224, 224), align_corners=False)
        target = F.interpolate(target, mode='bilinear', size=(224, 224), align_corners=False)
        input_vgg = self.vgg19_relu_5_2(input_)
        target_vgg = self.vgg19_relu_5_2(target)
        loss = F.l1_loss(input_vgg, target_vgg)
        return loss


class Color2EmbedLoss(nn.Module):
    def __init__(self, rank, lambda_reconstruction=1, lambda_perceptual=0.1):
        super(Color2EmbedLoss, self).__init__()
        self.lambda_reconstruction = lambda_reconstruction
        self.lambda_perceptual = lambda_perceptual
        self.reconstruction_loss = nn.SmoothL1Loss()
        self.perceptual_loss = PerceptualLoss(rank)

    def forward(self, pab, gtab, prgb, gtrgb):
        l_rec = self.reconstruction_loss(pab, gtab)
        l_per = self.perceptual_loss(prgb, gtrgb)
        return self.lambda_reconstruction * l_rec + self.lambda_perceptual * l_per, l_per, l_rec


if __name__ == '__main__':
    batch = 4
    pab = torch.rand(batch, 2, 256, 256)
    gtab = torch.rand(batch, 2, 256, 256)
    prgb = torch.rand(batch, 3, 256, 256)
    gtrgb = torch.rand(batch, 3, 256, 256)

    loss = Color2EmbedLoss()
    print(loss(pab, gtab, prgb, gtrgb))
    # print(mm(torch.rand(5, 3, 256, 256).to(0)).shape)
    # summary(loss.vgg19, (3, 224, 224))

