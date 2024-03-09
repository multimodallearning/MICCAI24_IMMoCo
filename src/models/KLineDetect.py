from fastmri.models import Unet


def get_unet(in_chans: int, out_chans: int, chans: int, num_pool_layers: int, drop_prob: float, **kwargs):
    return Unet(
        in_chans=in_chans,
        out_chans=out_chans,
        chans=chans,
        num_pool_layers=num_pool_layers,
        drop_prob=drop_prob,
        **kwargs,
    )