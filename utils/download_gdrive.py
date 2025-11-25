import gdown

url = 'https://drive.google.com/uc?id=1vbMFTe2E5OHp2h5o_YL-RUoIFKxsrZWT'
output = 'dataset.zip'
gdown.download(url, output, quiet=False)