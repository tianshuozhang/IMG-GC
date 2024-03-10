import requests
from PIL import Image
def get_input(processor,url,text=None,return_tensors="pt",local_img=False):
    if local_img:
        raw_image = Image.open(url).convert('RGB')
    else:    
        raw_image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    raw_image.show()
    if text is None:
        inputs = processor(raw_image,return_tensors="pt").to("cpu")
    else:
        inputs = processor(raw_image,text=text ,return_tensors="pt").to("cpu")
    return inputs

def get_img(url,local_img=False):
    if local_img:
        raw_image = Image.open(url).convert('RGB')
    else:    
        raw_image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    raw_image.show()
    return raw_image