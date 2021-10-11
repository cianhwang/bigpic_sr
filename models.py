from torch import nn
import torch
import common

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class EDSR(nn.Module):
    def __init__(self, conv=common.default_conv, num_channels=1, scale=1, n_post_blocks=0):
        super(EDSR, self).__init__()

        n_resblocks = 16
        n_feats = 64
        kernel_size = 3 
        act = nn.ReLU(True)

        # define head module
        m_head = [conv(num_channels, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        if scale == 1:
            m_tail = [
                conv(n_feats, 1, kernel_size)
            ]
        else:
            m_tail = [
                common.Upsampler(conv, scale, n_feats, act=False)
            ] + [
                common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=1
            ) for _ in range(n_post_blocks)
            ] + [
                conv(n_feats, 1, kernel_size)
            ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        return x 

if __name__=='__main__':
    inputs = torch.rand(4, 1, 64, 64).cuda()
    targets = torch.rand(4, 1, 256, 256).cuda()
    model = EDSR(num_channels=1, scale=4, n_post_blocks=3).cuda()
    print(model)
    preds = model(inputs)
    print(preds.size())
    loss = (targets - preds).sum()
    loss.backward()
