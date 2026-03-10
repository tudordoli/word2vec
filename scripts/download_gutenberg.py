import os
import urllib.request

ALICE_URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"


def main():
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "alice.txt")
    print("Downloading Alice in Wonderland...")
    urllib.request.urlretrieve(ALICE_URL, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
