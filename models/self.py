import torch
from torch import nn
def selfAttPatch(xq, xk, xv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_batchsize, C, width, height = xq.size()

    query_conv = nn.Conv2d(in_channels=C, out_channels=C // 8, kernel_size=1).to(device)
    key_conv = nn.Conv2d(in_channels=C, out_channels=C // 8, kernel_size=1).to(device)
    value_conv = nn.Conv2d(in_channels=C, out_channels=C, kernel_size=1).to(device)
    gamma = nn.Parameter(torch.zeros(1)).to(device)
    softmax = nn.Softmax(dim=-1)

    proj_query = query_conv(xq).view(m_batchsize, -1, width * height).permute(0, 2, 1)
    proj_key = key_conv(xk).view(m_batchsize, -1, width * height)
    energy = torch.bmm(proj_query, proj_key)
    attention = softmax(energy)
    proj_value = value_conv(xv).view(m_batchsize, -1, width * height)

    out = torch.bmm(proj_value, attention.permute(0, 2, 1))
    out = out.view(m_batchsize, C, width, height)
    out = gamma * out + xq.to(out.device)

    return out