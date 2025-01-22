from collections import Counter
from pathlib import Path
from string import punctuation as PUNCTUATS
import argparse
import re
import statistics

from lexical_diversity import lex_div
from nrclex import NRCLex
from readability import Readability
from syntok import segmenter
import nltk
import pandas as pd
import wordfreq


nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")

nltk.download("tagsets")
nltk.download("universal_tagset")


# os.environ["STANFORD_PARSER"] = "jar/stanford-parser.jar"
# os.environ["STANFORD_MODELS"] = "jar/stanford-parser-4.2.0-models.jar"


STOPWORDS_EN = set(nltk.corpus.stopwords.words("english"))
POS_TAGS = list(nltk.data.load("help/tagsets/upenn_tagset.pickle").keys())
NEGATIVES = ["anger", "fear", "disgust", "sadness", "negative"]
POSITIVES = ["anticipation", "trust", "surprise", "joy", "positive"]
EMOTIONS = NEGATIVES + POSITIVES
QUOTATS = "'\"‘’“”"


for _posi_ in range(len(POS_TAGS)):
    POS_TAGS[_posi_] = POS_TAGS[_posi_].replace(",", "，")
POS_TAGS.sort()


class Config(dict):
    """Copy from easydict.whl. Support nested dict."""

    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)
        for k in self.__class__.__dict__.keys():
            flag1 = k.startswith("__") and k.endswith("__")
            flag2 = k in ("fromfile", "update", "pop")
            if any([flag1, flag2]):
                continue
            setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        super(Config, self).__setattr__(name, value)
        super(Config, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(Config, self).pop(k, d)


def flatten_dict(d, parent_key="", sep="/"):
    """Flattens a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def stack_dict(ds: list):
    """Suppose ds are already flattened."""
    keys = list(ds[0].keys())
    assert all(keys == list(_.keys()) for _ in ds[1:])
    d2 = {k: [] for k in keys}
    for d in ds:
        for k, v in d.items():
            d2[k].append(v)
    return d2


def count_text_statistics(parset, text, stat):
    tokens_sent = nltk.sent_tokenize(text)
    tokens_word0 = nltk.word_tokenize(text)
    text_segment = "\n\n".join(
        "\n".join(" ".join(t.value for t in s) for s in p)
        for p in segmenter.analyze(text)
    )

    stat.num_token_all = len(tokens_word0)

    tokens_word = [_ for _ in tokens_word0 if _ not in PUNCTUATS]
    stat.num_word = len(tokens_word)

    lemmas = lex_div.flemmatize(text)
    ttr = lex_div.ttr(lemmas)
    stat.ttr = ttr

    def syntax_depth(bad_chars=["%"]):
        """
        The parser needs to be initialised in order for the code to work.
        Navigate to the file location where the Core NLP parser is and run this command
        java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000 &
        """
        depth = []
        depth_vp = []
        depth_np = []
        for sentence in tokens_sent:
            for i in bad_chars:
                sentence = sentence.replace(i, "")
            p = list(parset.parse(sentence.split()))
            root = p[0]
            dvp = []
            dnp = []
            for phrase in root.subtrees(lambda _: _.label() == "S"):
                for fraasi in phrase.subtrees(lambda _: _.label() == "VP"):
                    dvp.append(fraasi.height())
                for fraasi in phrase.subtrees(lambda _: _.label() == "NP"):
                    dnp.append(fraasi.height())
                if len(dvp) > 0:
                    depth_vp.append(max(dvp))
                if len(dnp) > 0:
                    depth_np.append(max(dnp))
            depth.append(p[0].height())

        if len(depth) < 1:
            median = "NA"
        else:
            median = statistics.median(depth)
        stat.med_syntax_depth = median

        if len(depth_vp) < 1:
            median_vp = "NA"
        else:
            median_vp = statistics.median(depth_vp)
        stat.med_vp_depth = median_vp

        if len(depth_np) < 1:
            median_np = "NA"
        else:
            median_np = statistics.median(depth_np)
        stat.med_np_depth = median_np

    def words_per_sentence():
        num_sentence = len(tokens_sent)
        assert num_sentence > 0
        stat.num_sentence = num_sentence
        words_per_sentence = len(tokens_word) / num_sentence
        stat.num_word_per_sentence = words_per_sentence

    def readable():
        r = Readability(text_segment)
        if len(tokens_word) < 100:
            stat.f_score = "NA"  # TODO XXX
            stat.g_score = "NA"  # TODO XXX
        else:
            fk = r.flesch_kincaid()
            stat.f_score = fk.score
            gf = r.gunning_fog()
            stat.g_score = gf.score
        if len(tokens_sent) >= 30:
            s = r.smog()
            stat.s_score = s.score
        else:
            stat.s_score = "NA"  # TODO XXX

    def word_length():
        num_word = len(tokens_word)
        assert num_word > 0
        num_char = sum(len(_) for _ in tokens_word)
        chars_per_word = num_char / num_word
        stat.avg_word_len = chars_per_word

    def nrc_emotions():
        emotion_cnt = Counter()
        for word in tokens_word:
            nrclex_word = NRCLex(word)
            for key, value in nrclex_word.raw_emotion_scores.items():
                if key in EMOTIONS:
                    emotion_cnt[key] += value
        stat.num_emotion.update(emotion_cnt)

        nrclex_text = NRCLex(text)
        neg = 0
        pos = 0
        for key, value in nrclex_text.affect_frequencies.items():
            if key in NEGATIVES:
                neg += value
            elif key in POSITIVES:
                pos += value
        stat.negative_intensity = neg
        stat.positive_intensity = pos

    def word_tokens():
        num_caps = len(
            [_ for _ in tokens_word if re.match(r"\b[A-Z]+\b", _) and len(_) > 1]
        )
        stat.num_cap = num_caps

        num_punctuat = len([_ for _ in tokens_word0 if _ in PUNCTUATS])
        stat.num_punctuat = num_punctuat

        num_quotat = len([_ for _ in tokens_word0 if _ in QUOTATS])
        stat.num_quotat = num_quotat / 2

        pos_tags = nltk.pos_tag(tokens_word)
        pos_cnt = Counter()
        for pair in pos_tags:
            pos_cnt[pair[1]] += 1
        stat.num_pos.update(pos_cnt)

    def stop_words():
        num_stopword = len([_ for _ in tokens_word if _ in STOPWORDS_EN])
        stat.num_stopword = num_stopword

    def fluency():
        freqs = []
        for word in tokens_word:
            freq = wordfreq.word_frequency(word, "en")
            if freq > 0:
                freqs.append(freq)
        freq_avg = sum(freqs) / len(tokens_word)
        stat.word_freq = freq_avg

        # average frequency of all words in article
        freqs = sorted(freqs)
        stat.word_freq_mc9 = sum(freqs[-9:]) / 9
        stat.word_freq_lc3 = sum(freqs[:3]) / 3  # least common 3 words in article

    words_per_sentence()
    word_length()
    readable()
    nrc_emotions()
    word_tokens()
    stop_words()
    syntax_depth()
    fluency()


def prepare_text_sample(original_text):
    texts = original_text.split("\n\n\n\n\n")
    assert len(texts) == 6
    for i in range(len(texts)):
        text0 = texts[i]
        text = [_.strip() for _ in text0.split("\n")]
        text = [_ for _ in text if len(_) > 0]
        text = [_ for _ in text if not (_.startswith("*") and _.endswith("*"))]
        texts[i] = "\n".join(text)
    sources = [
        "question",
        "human",
        "chatgpt",
        "chatgpt_imitat",
        "doubao",
        "doubao_imitat",
    ]
    writing = Config(dict(zip(sources, texts)))
    return writing


def process_one_sample(nlp_server, text_file, stat_file):
    parset = nltk.parse.CoreNLPParser(nlp_server)

    with open(text_file, "r", encoding="utf-8") as fp:
        original_text = fp.read()
    sample = prepare_text_sample(original_text)

    stats = []

    for source, text in sample.items():
        if source == "question":
            continue

        stat = Config({})
        # stylistic
        stat.num_sentence = 0
        stat.num_token_all = 0
        stat.num_word = 0
        stat.num_word_per_sentence = 0
        stat.num_punctuat = 0
        stat.num_quotat = 0
        stat.num_cap = 0
        stat.num_stopword = 0
        stat.num_pos = Config({t: 0 for t in POS_TAGS})
        # complexity
        stat.avg_word_len = 0
        stat.ttr = 0
        stat.med_syntax_depth = 0
        stat.med_vp_depth = 0
        stat.med_np_depth = 0
        stat.f_score = 0
        stat.g_score = 0
        stat.s_score = 0  # TODO XXX ???
        stat.word_freq = 0
        stat.word_freq_mc9 = 0
        stat.word_freq_lc3 = 0
        # sentiment
        stat.num_emotion = Config({e: 0 for e in EMOTIONS})
        stat.negative_intensity = 0
        stat.positive_intensity = 0

        count_text_statistics(parset, text, stat)
        stat = flatten_dict(stat)

        stats.append(stat)

    df = pd.DataFrame(stack_dict(stats))
    print(df.T)
    df.T.to_csv(f"{stat_file}.csv")


def parse_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.nlp_server = "http://192.168.1.102:9000"
    args.data_path = Path("d:\Active\LinyaoDu\ielts-human-chatgptx2-doubaox2-imitat")
    args.save_path = Path("stat")

    if not args.save_path.exists():
        args.save_path.mkdir()

    files = args.data_path.glob("*.txt")
    for file in files:
        filename = file.name
        print(f"processing texts in file ``{filename}``...")

        text_file = args.data_path / filename
        stat_file = args.save_path / filename
        process_one_sample(args.nlp_server, text_file, stat_file)


if __name__ == "__main__":
    main()
