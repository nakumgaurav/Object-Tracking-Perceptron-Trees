from PIL import Image
import os

sequence = 'ball'
image_dir = 'frames/' + sequence
def downsample(img_name):
    img = Image.open(os.path.join(image_dir,img_name))
    a = img.size
    a = [int(b/2) for b in a ]
    resized = img.resize(a, Image.LANCZOS)
    return resized

files = os.listdir(image_dir)
for fname in files:
    if('.png' in fname or '.jpg' in fname or '.jpeg' in fname):
        img_ds = downsample(fname)
        img_ds.save(os.path.join('new_frames/' + sequence, fname))