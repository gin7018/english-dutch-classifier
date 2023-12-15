import re

import pandas as pd

import common


def remove_special_characters(w):
    return re.sub(r'[^\w\s]', "", w)


def most_common_words_in_language(df, language):
    df = df[df[common.CLASS] == language]
    word_frequency_table = {}

    for idx, row in df.iterrows():
        words = row["sentence"].split(" ")

        for word in words:
            word = word.lower()
            word = remove_special_characters(word)

            if len(word) == 0:
                continue

            if word not in word_frequency_table:
                word_frequency_table[word] = 0
            word_frequency_table[word] += 1

    sorted_frequency_table = sorted(word_frequency_table.items(), key=lambda e: e[1], reverse=True)

    # for it in sorted_frequency_table[:10]:
    #     print(it)
    words = list(map(lambda x: x[0], sorted_frequency_table[:10]))
    return words


def most_frequent_letters_in_language(df):
    letter_frequency_table = {}

    for idx, row in df.iterrows():
        letters = list(row["sentence"])

        for letter in letters:
            letter = letter.lower()

            if not letter.isalpha():
                continue

            if letter not in letter_frequency_table:
                letter_frequency_table[letter] = 0
            letter_frequency_table[letter] += 1

    sorted_frequency_table = sorted(letter_frequency_table.items(), key=lambda e: e[1], reverse=True)

    for it in sorted_frequency_table[:10]:
        print(it)


def average_word_length(df, language):
    df = df[df[common.CLASS] == language]

    number_of_words = 0
    total_word_length = 0

    for idx, row in df.iterrows():
        words = row["sentence"].split(" ")

        for word in words:
            word = word.lower()
            word = remove_special_characters(word)

            if len(word) == 0:
                continue

            total_word_length += len(word)
            number_of_words += 1

    return total_word_length / number_of_words


def average_sentence_word_length(sentence):
    words = sentence.split(" ")

    number_of_words = 0
    total_word_length = 0

    for word in words:
        word = word.lower()
        word = remove_special_characters(word)

        if len(word) == 0:
            continue

        total_word_length += len(word)
        number_of_words += 1
    return total_word_length / number_of_words


def average_unique_characters_in_a_word(df):
    number_of_words = 0
    total_unique_letter = 0

    for idx, row in df.iterrows():
        words = row["sentence"].split(" ")

        for word in words:
            word = word.lower()
            word = remove_special_characters(word)

            if len(word) == 0:
                continue

            total_unique_letter += len(set(list(word))) / len(word)
            number_of_words += 1

    return total_unique_letter / number_of_words


def create_features_for_training(df):
    for f in provide_features():
        df[f] = False

    common_en_words = most_common_words_in_language(df, common.CLASS_A)
    common_nl_words = most_common_words_in_language(df, common.CLASS_B)

    too_common = []
    for w in common_en_words:
        if w in common_nl_words:
            too_common.append(w)

    common_en_words = list(filter(lambda ww: ww not in too_common and len(ww) != 1, common_en_words))
    common_nl_words = list(filter(lambda ww: ww not in too_common and len(ww) != 1, common_nl_words))

    average_dutch_word_len = average_word_length(df, common.CLASS_A)
    average_english_word_len = average_word_length(df, common.CLASS_B)
    midpoint = (average_dutch_word_len + average_english_word_len) / 2

    for idx, row in df.iterrows():
        sentence = row["sentence"].split(" ")
        cleaned = list(map(lambda ww: remove_special_characters(ww).lower(), sentence))

        df.loc[idx, "ends_with_ed"] = any(w.endswith("ed") for w in cleaned)
        df.loc[idx, "ends_with_ly"] = any(w.endswith("ly") for w in cleaned)
        df.loc[idx, "ends_with_ng"] = any(w.endswith("ng") for w in cleaned)

        df.loc[idx, "has_ij"] = any("ij" in w for w in cleaned)
        df.loc[idx, "has_oe"] = any("oe" in w for w in cleaned)
        df.loc[idx, "has_sch"] = any("sch" in w for w in cleaned)

        df.loc[idx, "common_en_words"] = any(w in common_en_words for w in cleaned)
        df.loc[idx, "common_nl_words"] = any(w in common_nl_words for w in cleaned)

        df.loc[idx, "dutch_word_len"] = average_sentence_word_length(row["sentence"]) < midpoint
        df.loc[idx, "english_word_len"] = average_sentence_word_length(row["sentence"]) >= midpoint

    df["weight"] = 1 / df.shape[0]

    with open("features.txt", "w") as f:
        for w in common_en_words:
            f.write(w + " ")
        f.write("\n")
        for w in common_nl_words:
            f.write(w + " ")
        f.write("\n")
        f.write(str(midpoint))


def load_features_for_predicting(df):
    for f in provide_features():
        df[f] = False

    with open("/autograder/submission/features.txt") as f:
        common_en_words = [str(ww) for ww in f.readline().strip().split(" ")]
        common_nl_words = [str(ww) for ww in f.readline().strip().split(" ")]
        word_len_threshold = float(f.readline().strip())

    for idx, row in df.iterrows():
        sentence = row["sentence"].split(" ")
        cleaned = list(map(lambda ww: remove_special_characters(ww).lower(), sentence))

        df.loc[idx, "ends_with_ed"] = any(w.endswith("ed") for w in cleaned)
        df.loc[idx, "ends_with_ly"] = any(w.endswith("ly") for w in cleaned)
        df.loc[idx, "ends_with_ng"] = any(w.endswith("ng") for w in cleaned)

        df.loc[idx, "has_ij"] = any("ij" in w for w in cleaned)
        df.loc[idx, "has_oe"] = any("oe" in w for w in cleaned)
        df.loc[idx, "has_sch"] = any("sch" in w for w in cleaned)

        df.loc[idx, "common_en_words"] = any(w in common_en_words for w in cleaned)
        df.loc[idx, "common_nl_words"] = any(w in common_nl_words for w in cleaned)

        df.loc[idx, "dutch_word_len"] = average_sentence_word_length(row["sentence"]) < word_len_threshold
        df.loc[idx, "english_word_len"] = average_sentence_word_length(row["sentence"]) >= word_len_threshold

    df["weight"] = 1 / df.shape[0]


def provide_features():
    features = ["ends_with_ed", "ends_with_ly", "ends_with_ng", "has_ij",
                "has_oe", "has_sch", "common_en_words", "common_nl_words",
                "dutch_word_len", "english_word_len"]
    return features


def main():
    dataset = pd.read_csv("text/text_material.csv")

    # english_sentences = dataset[dataset[common.CLASS] == common.CLASS_A]
    # dutch_sentences = dataset[dataset[common.CLASS] == common.CLASS_B]
    create_features_for_training(dataset)
    load_features_for_predicting(dataset)

    # print("top 10 most common english words:")
    # most_common_words_in_language(english_sentences)
    # print()
    # print("top 10 most common Dutch words:")
    # most_common_words_in_language(dutch_sentences)

    # print("top 10 most common english letters:")
    # most_frequent_letters_in_language(english_sentences)
    # print()
    # print("top 10 most common Dutch letters:")
    # most_frequent_letters_in_language(dutch_sentences)

    # print(f"average english word length: {average_word_length(english_sentences)}")
    # print(f"average dutch word length: {average_word_length(dutch_sentences)}")

    # print(f"average english unique letters / word: {average_unique_characters_in_a_word(english_sentences)}")
    # print(f"average dutch unique letters / word: {average_unique_characters_in_a_word(dutch_sentences)}")


if __name__ == '__main__':
    main()
