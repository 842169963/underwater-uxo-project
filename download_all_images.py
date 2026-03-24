import os
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

START_URL = "https://www2.in.tu-clausthal.de/uxo-detection/Data_Images_Videos/Image/Images/"
SAVE_ROOT = r"C:\Users\stephenxxy\Desktop\project\uxo_project\tuc_images\Images"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")
headers = {"User-Agent": "Mozilla/5.0"}
visited = set()


def is_image_file(name: str) -> bool:
    return name.lower().endswith(IMG_EXTS)


def download_file(url: str, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        print(f"[SKIP] {save_path}")
        return

    try:
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"[OK]   {save_path}")
    except Exception as e:
        print(f"[FAIL] {url} -> {e}")


def crawl(url: str, local_dir: str):
    if url in visited:
        return
    visited.add(url)

    print(f"[CRAWL] {url}")

    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"[FAIL PAGE] {url} -> {e}")
        return

    soup = BeautifulSoup(r.text, "html.parser")

    for a in soup.find_all("a"):
        href = a.get("href")
        text = a.get_text(strip=True)

        if not href:
            continue

        if href == "../" or "Parent Directory" in text:
            continue

        full_url = urljoin(url, href)

        # 只允许在 Images/ 下面递归
        if not full_url.startswith(START_URL):
            print(f"[SKIP OUTSIDE] {full_url}")
            continue

        if href.endswith("/"):
            sub_name = href.strip("/")
            if not sub_name:
                continue
            sub_dir = os.path.join(local_dir, sub_name)
            crawl(full_url, sub_dir)

        elif is_image_file(href):
            save_path = os.path.join(local_dir, href)
            download_file(full_url, save_path)

        else:
            print(f"[SKIP NON-IMG] {full_url}")


def main():
    os.makedirs(SAVE_ROOT, exist_ok=True)
    crawl(START_URL, SAVE_ROOT)
    print("Done.")


if __name__ == "__main__":
    main()