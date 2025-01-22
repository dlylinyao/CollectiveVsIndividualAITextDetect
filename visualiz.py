from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def draw_multiple_values(labels, keys, values):
    assert (len(labels), len(keys)) == values.shape

    max_base = np.max(values, 0, keepdims=True)
    max_base[max_base == 0] = 1
    values = np.array(values) / max_base  # normalize

    plt.vlines(range(len(keys)), 0, 1, colors="grey", linewidth=0.5, alpha=0.9)
    for lbl, val in zip(labels, values):
        plt.scatter(range(len(val)), val, label=f"{lbl}")

    plt.legend()
    plt.xticks(range(len(keys)), keys, rotation=90)


def main_compare_different_texts_metrics(
    stat_file=Path("ielts_5_test_1.txt.csv"),
    viz_file=None,
    labels=["human", "chatgpt", "chatgpt_rewrite", "doubao", "doubao_rewrite"],
):
    text = stat_file.read_text(encoding="utf-8")
    lines = text.strip().split("\n")[1:]  # remove first line
    lines = list(map(lambda _: _.strip().split(","), lines))
    assert all(len(_) == len(labels) + 1 for _ in lines)

    ks = []
    vs = []
    for line in lines:
        k = line[0]
        v = line[1:]
        for i in range(len(v)):
            if "NA" == v[i]:
                v[i] = float("inf")
            else:
                v[i] = float(v[i])
        ks.append(k)
        vs.append(v)

    vs = np.array(vs).transpose()

    plt.figure(figsize=[15, 5])
    draw_multiple_values(labels, ks, vs)
    plt.tight_layout()
    if viz_file:
        plt.savefig(viz_file)
    # plt.show()


def main():
    stat_path = Path("stat")
    save_path = Path("viz")
    if not save_path.exists():
        save_path.mkdir()
    stat_files = list(stat_path.glob("*.csv"))
    stat_files.sort()
    for stat_file in stat_files:
        print(f"visualizing statistics in file ``{str(stat_file)}``")
        viz_file = save_path / f"{stat_file.name.split('.')[0]}.png"
        main_compare_different_texts_metrics(stat_file, viz_file)


if __name__ == "__main__":
    # main_compare_different_texts_metrics()
    main()
