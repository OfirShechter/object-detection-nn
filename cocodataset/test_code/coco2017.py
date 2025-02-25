#%%
from pycocotools.coco import COCO
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt

#%%
# print current working directory
import os
os.getcwd()
# %%
# Load the COCO dataset
coco = COCO('../annotations/instances_train2017.json')
# %%
# Get all category IDs (e.g., person, car, etc.)
cat_ids = coco.getCatIds(catNms=['dog', 'person'])  # Change to any category
img_ids = coco.getImgIds(catIds=cat_ids)

# %%
cat_ids, img_ids
# %%
# Load and display an image
img_info = coco.loadImgs(img_ids[np.random.randint(0, len(img_ids))])[0]
img_url = img_info['coco_url']
img_info
# %%
def load_image_from_url(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        img_arr = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        return img
    else:
        print("Failed to download image")
        return None

# Load and show the image
img = load_image_from_url(img_url)

if img is not None:
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# %%
# Load annotations for the selected image
ann_ids = coco.getAnnIds(imgIds=img_info['id'], catIds=cat_ids, iscrowd=False)
annotations = coco.loadAnns(ann_ids)
annotations[0].keys()

#%%
# Draw bounding boxes
for ann in annotations:
    x, y, w, h = ann['bbox']
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

# Display the image with bounding boxes
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# %%
