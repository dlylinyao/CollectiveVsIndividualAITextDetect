import matplotlib.pyplot as plt


def draw_accs_mean_std(figsize=[6, 3], accs=dict(), save_file=None):
    plt.figure(figsize=figsize)

    for k, [mean, std] in accs.items():
        bar = plt.bar(k, mean, yerr=std)
        assert len(bar) == 1
        bar = bar[0]

        x = bar.get_x() + bar.get_width() / 2
        y = 0.1
        plt.text(
            x,
            y,
            f"{mean:.4f}+/-{std:.4f}",
            color="white",
            ha="center",
            va="top",
            fontsize=10,
        )

    plt.tight_layout()
    plt.ylim([-0.1, 1.1])
    if save_file:
        plt.savefig(save_file)

    plt.show()


def main():
    draw_accs_mean_std(
        accs=dict(proposed=[0.9425, 0.0240], baseline=[0.9487, 0]),
        save_file="acc_pos_neg.png",
    )
    draw_accs_mean_std(
        accs=dict(proposed=[0.8024, 0.0554], baseline=[0.3397, 0]),
        save_file="acc_pos_imi.png",
    )
    draw_accs_mean_std(
        accs=dict(proposed=[0.8389, 0.0249], baseline=[0.6000, 0]),
        save_file="acc_all.png",
    )


if __name__ == "__main__":
    main()
