import torch


def corr(x1, x2, kernel_radius):
    """
    :param x1: Tensor of shape [b, c, h, w]
    :param x2: Tensor of shape [b, c, h, w] and have the same shape of x1
    :param kernel_radius: kernel_radius
    :return: Tensor after correlation operation, shape of [b, kernel_size, kernel_size, h, w]
    """
    assert x1.shape == x2.shape

    b, c, h, w = x1.shape
    kernel_size = 2 * kernel_radius + 1

    volumn = x1.new_zeros([b, kernel_size, kernel_size, h, w])
    for i in range(-1 * kernel_radius, kernel_radius + 1):
        idxi = -1 * i + kernel_radius
        for j in range(-1 * kernel_radius, kernel_radius + 1):
            idxj = -1 * j + kernel_radius

            clip_x1 = [(slice(d, None, None) if d >= 0 else slice(None, d, None)) for d in (i, j)]
            clip_x2 = [(slice(None, -d, None) if d > 0 else slice(-d, None, None)) for d in (i, j)]
            clip_tuple_x1 = (slice(None, None, None), slice(None, None, None), *clip_x1)
            clip_tuple_x2 = (slice(None, None, None), slice(None, None, None), *clip_x2)

            clip_tuple_volumn = (slice(None, None, None), idxi, idxj, *clip_x1)

            volumn[clip_tuple_volumn] = torch.sum(x1[clip_tuple_x1] * x2[clip_tuple_x2], dim=1)
    return volumn


if __name__ == "__main__":
    _x1 = torch.tensor([[x for x in range(1, 5)] for y in range(1, 5)])[None][None].float().to(torch.device("cuda"))
    _x2 = torch.tensor([[y for x in range(1, 5)] for y in range(1, 5)])[None][None].float().to(torch.device("cuda"))

    _x1.requires_grad_(True)

    _y = corr(_x1, _x2, kernel_radius=1)
    _y[0][1][1][3][2].backward()
    print(_x1.grad)
    pass
