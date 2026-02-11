import os
from icrawler.builtin import BingImageCrawler

def download_images(keyword, folder, max_num=1000):
    os.makedirs(folder, exist_ok=True)
    
    crawler = BingImageCrawler(storage={'root_dir': folder})
    crawler.crawl(
        keyword=keyword,
        max_num=max_num,
        filters=None
    )

if __name__ == "__main__":
    # Training images
    download_images("dashiki clothing", "dataset/train/dashiki", 1000)
    download_images("indian kurti", "dataset/train/kurti", 1000)

    # Validation images
    download_images("dashiki african shirt", "dataset/val/dashiki", 200)
    download_images("traditional kurti", "dataset/val/kurti", 200)

    # Test images
    download_images("dashiki outfit", "dataset/test/dashiki", 200)
    download_images("women kurti", "dataset/test/kurti", 200)
