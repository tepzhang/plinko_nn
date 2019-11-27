import torch
import torch.nn.functional as F


def merge_dicts(a: dict, b: dict):
    d = {}
    for k, v in a.items():
        d[k] = v
    for k, v in b.items():
        d[k] = v
    return d


def drop_constants(df):
    """
    Drops columns where all the values are the same
    :param df:
    :return:
    """
    return df.loc[:, (df != df.iloc[0]).any()]


def expand_along_dim(tensor, n, dim):
    """
    Expands a tensor n times along dimension dim.
    :param tensor:
    :param n:
    :param dim:
    :return:
    """
    expanded = tensor.expand(n, *tensor.shape)
    dim_order = list(range(1, len(tensor.shape)+1))
    dim_order.insert(dim, 0)
    return expanded.permute(dim_order)


def normalize(x, dim=None):
    dim = len(x.shape)-1 if dim is None else dim
    if x.dtype != torch.double:
        x = x.double()
    return x/expand_along_dim(torch.sum(x, dim=dim), x.shape[dim], dim)


def select_1d_from_2d(tensor, indices):
    """
    :param tensor: 2d tensor of shape (a, b)
    :param indices: 1d int tensor of length a
    :return: 1d vector where vector[i] = tensor[i, indices[i]]
    """
    return tensor.gather(1, indices.view(-1, 1)).squeeze()


def select_along_dim(tensor, indices):
    """
    tensor: tensor with shape [d_1, ..., d_k, ..., d_n]
    indices: tensor with shape [d_1, ..., d_k]
    output: tensor with shape [d_1, ..., d_(k-1), d_(k+1), ..., d_n]
    >>> a = torch.arange(2*3*4*5).view(2,3,5,4)
    >>> b = torch.tensor([[0, 0, 0], [3,2,1]])
    >>> select_along_dim(a, b)
    tensor([[[  0,   1,   2,   3],
             [ 20,  21,  22,  23],
             [ 40,  41,  42,  43]],

            [[ 72,  73,  74,  75],
             [ 88,  89,  90,  91],
             [104, 105, 106, 107]]])
    """
    dim = len(indices.shape)
    while len(indices.shape) < len(tensor.shape):
        indices = indices.unsqueeze(-1)
    shape = (-1,)*(dim+1) + tensor.shape[dim+1:]
    indices = indices.expand(shape)
    return tensor.gather(dim, indices).squeeze(dim)


def log_sum_exp(x, dim=0):
    """
    Compute the log(sum(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which sum is computed

    Return:
        _: tensor: (...): log(sum(exp(x), dim))
    """
    max_x = torch.max(x, dim)[0]
    new_x = x - max_x.unsqueeze(dim).expand_as(x)
    return max_x + (new_x.exp().sum(dim)).log()


def identity_predictor_check(model, dataloader):
    """
    Used to perform a sanity check on the model by comparing the outputs to
    the inputs and targets. This is to ensure that the model doesn't simply
    learn an identity function to reduce MSE since positions at times t and t+1
    are fairly close together.

    mse_input: the mean-squared error to the input positions (time t)
    mse_target: the mean-squared error to the target positions (time t+1)

    Currently to be specifically used with GRUPredictor.
    :param model: GRUPredictor
    :param dataloader: DataLoader
    :return: mse_input, mse_target
    """
    mse_input = 0
    mse_target = 0
    with torch.no_grad():
        for batch_i, batch in enumerate(dataloader):
            gm = model(batch['envs'], batch['states'], 0)
            gm_mu = gm.mu[..., 0, :]  # gets mu of first gaussian
            mse_input += F.mse_loss(gm_mu, batch['states'])
            mse_target += F.mse_loss(gm_mu, batch['targets'])
    return mse_input, mse_target
