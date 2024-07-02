import numpy as np
import torch
import random
import jiwer
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pylab as plt



def wer(a, b):
    a = jiwer.RemovePunctuation()(a)
    b = jiwer.RemovePunctuation()(b)
    return jiwer.wer(a, b, reference_transform=jiwer.wer_standardize, hypothesis_transform=jiwer.wer_standardize)


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class AttentionVisualizer(object):
    def __init__(self, figsize=(32, 16)):
        self.figsize = figsize

    def plot(self, info, return_axes=False):
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)

        # clip range
        if "clip_range" in info:
            info["attn"] = np.clip(info["attn"], -info["clip_range"], info["clip_range"])
            cax = ax.matshow(info["attn"])
            cax.set_clim(vmin=-info["clip_range"], vmax=info["clip_range"])
        else:
            cax = ax.matshow(info["attn"])
        fig.colorbar(cax, ax=ax)

        for i in np.arange(len(info["x_labels"])):
            for j in np.arange(len(info["y_labels"])):
                ax.text(j, i, f"{info['attn'][i][j]:.2f}", ha="center", va="center")

        ax.set_title(info["title"], fontsize=28)
        ax.set_xticks(np.arange(len(info["x_labels"])))
        ax.set_xticklabels(info["x_labels"], rotation=90, fontsize=8)
        ax.set_yticks(np.arange(len(info["y_labels"])))
        ax.set_yticklabels(info["y_labels"], fontsize=8)

        if 'x_label' in info:
            ax.set_xlabel(info['x_label'])
        if 'y_label' in info:
            ax.set_ylabel(info['y_label'])

        if return_axes:
            return fig, ax
        return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def unwrap_loss(loss_data):
    """
    Transform nested list of dict into dict of nested list, all dicts should have same keys.
    This is a recursive implementation.
    """
    def _unwrap(data: list):
        res = {}
        for x in data:
            if isinstance(x, list):  # not last level
                unwrapped_x = _unwrap(x)
            else:
                unwrapped_x = x
            for key in unwrapped_x:
                if key not in res:
                    res[key] = []
                res[key].append(unwrapped_x[key])
        return res
    return _unwrap(loss_data)
