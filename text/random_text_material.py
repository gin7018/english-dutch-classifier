import requests
from bs4 import BeautifulSoup
import pandas as pd


session = requests.session()

LANGUAGES = {
    "english": "https://en.wikipedia.org/w/api.php",
    "dutch": "https://nl.wikipedia.org/w/api.php"
}

HTTP_PARAMETERS = {
    "action": "query",
    "format": "json",
    "list": "random",
    "rnlimit": "10",
    "rnnamespace": "0",
    "prop": "revisions",
    "rvprop": "content"
}


def get_random_wikipedia_articles(language):
    response = session.get(url=LANGUAGES[language], params=HTTP_PARAMETERS)
    data = response.json()

    random_articles = data["query"]["random"]
    return random_articles


def get_article_content(language, page_id):
    content_parameters = {
        "action": "query",
        "format": "json",
        "pageids": page_id,
        "prop": "extracts",
        "rvprop": "content"
    }
    content_response = session.get(url=LANGUAGES[language], params=content_parameters)
    data = content_response.json()

    extract = data["query"]["pages"][str(page_id)]["extract"]
    soupy = BeautifulSoup(extract, "html.parser").decode("utf-8")
    resulting_text = soupy

    return resulting_text


def read_15_words_to_language_db(text_material, language):
    text_material = text_material.split("\n")

    def longer_than_15(sentence):
        return len(sentence.split(" ")) >= 15
    text_material = list(filter(longer_than_15, text_material))

    with open(f"{language}_db.txt", "a", encoding="utf-8") as en_f:
        for text in text_material:
            en_f.writelines(text + "\n")


def read_10_articles(articles, language):
    for article in articles:
        content = get_article_content(language, article["id"])
        read_15_words_to_language_db(content, language)


def provide_db():
    df = pd.DataFrame(columns=["sentence", "class"])

    with open("english_db.txt", encoding="utf-8") as en_f:
        for line in en_f:
            words = line.split(" ")
            words = list(filter(lambda w: w != " ", words))
            for i in range(0, len(words), 15):
                if i+15 >= len(words):
                    break
                sentence = " ".join(words[i:i+15])
                df.loc[len(df)] = {"sentence": sentence, "class": "en"}

    with open("dutch_db.txt", encoding="utf-8") as ln_f:
        for line in ln_f:
            words = line.split(" ")
            words = list(filter(lambda w: w != " ", words))
            for i in range(0, len(words), 15):
                if i+15 >= len(words):
                    break
                sentence = " ".join(words[i:i+15])
                df.loc[len(df)] = {"sentence": sentence, "class": "nl"}

    df.to_csv("text/text_material.csv")
    # return df


def main():
    # articles = get_random_wikipedia_articles("english")
    # read_10_articles(articles, "english")

    # articles = get_random_wikipedia_articles("dutch")
    # read_10_articles(articles, "dutch")

    # df = provide_db()
    provide_db()

    print("done done")


if __name__ == '__main__':
    main()
