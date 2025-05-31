import requests
import os

def download_vietnamese_stopwords(filename="data/download/vietnamese_stopwords.txt"):
    try:
        if len(load_vietnamese_stopwords(filename)) > 0:
            print("Vietnamese stopwords already downloaded.")
            return True
    except Exception as e:
        print(f"Error when checking for vietnamese stopwords: {e}")
    
    try:
        url = "https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords.txt"
        response = requests.get(url)
        vietnamese_stopwords = response.text.splitlines()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as file:
            for word in vietnamese_stopwords:
                file.write(word + "\n")
        print(f"Vietnamese stopwords downloaded and saved to {filename}")
        return True
    except Exception as e:
        print(f"Error when downloading vietnamese stopwords: {e}")
        return False
def load_vietnamese_stopwords(filename="data/download/vietnamese_stopwords.txt"):
    with open(filename, "r", encoding="utf-8") as file:
        vietnamese_stopwords = file.read().splitlines()
    return vietnamese_stopwords

if __name__ == "__main__":
    vietnamese_stopwords = download_vietnamese_stopwords()
    print(vietnamese_stopwords)