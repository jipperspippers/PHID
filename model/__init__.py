from .phidnet import PHID

"""Define commonly used architecture"""
def  phid():
    net = PHID(in_channels=1, out_channels=1, num_features=64, dwt=3)
    net.use_2dconv = False
    net.bandwise = False
    return net
    