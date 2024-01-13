import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pylab as plt


class AttentionVisualizer(object):
    def __init__(self, figsize=(32, 16)):
        self.figsize = figsize

    def plot(self, info):
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)
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

        return fig


def main(result_path, output_path):
    visualizer = AttentionVisualizer()
    male_spks = ["3005", "6432", "8131", "4198", "7105"]
    female_spks = ["5484", "6070", "5442", "3528", "8280"]
    spks = male_spks + female_spks
    x_labels = y_labels = spks

    baseline = np.zeros(len(spks))
    data = np.zeros((len(spks), len(spks)))
    with open(result_path, 'r') as f:
        for line in f:
            a, b, c = line.strip().split('|')
            if a == "None":
                baseline[spks.index(b)] = float(c)
            else:
                data[spks.index(a)][spks.index(b)] = float(c)
    info = {
        "title": "WER Improvement",
        "x_labels": x_labels,
        "y_labels": y_labels,
        "x_label": "tgt speaker",
        "y_label": "src speaker",
        "attn": -(data - baseline)
    }
    fig = visualizer.plot(info)
    plt.savefig(output_path)


if __name__ == "__main__":
    main("spk_results.txt", "spk_results.jpg")
    main("spk_results-n.txt", "spk_results-n.jpg")
    # main("spk_results_iid_adapt.txt", "spk_results_iid_adapt.jpg")
    # main("spk_results_iid_adapt-matching.txt", "spk_results_iid_adapt-matching.jpg")
