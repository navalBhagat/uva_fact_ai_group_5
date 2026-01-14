import os
import numpy as np
from PIL import Image
from torchvision import datasets

# Since the code uses an older version of torchvision, the link to EMNIST is
# broken. We need to manually change the url and filename parameters to make
# the automatic download work.
datasets.EMNIST.url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
datasets.EMNIST.filename = "gzip.zip"


class EMNISTBase(datasets.EMNIST):
    def __init__(self, datadir, size=None, split="digits", **kwargs):
        self.size = size
        self.split = split
        # On Snellius, it is preferred to store the data in scratch-local,
        # which is found at the $TMPDIR variable.
        cachedir = os.environ.get("TMPDIR", os.path.expanduser("~/.cache"))
        super().__init__(root=os.path.join(cachedir, datadir), split=split, **kwargs)

    def __getitem__(self, index):
        image, label = super().__getitem__(index)

        img = np.array(image).astype(np.uint8)
        img_three_channels = np.stack((img, img, img), axis=2)

        img = img_three_channels

        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size))

        image = np.array(image).astype(np.uint8)

        example = {}
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["class_label"] = label

        return example


class EMNISTTrain(EMNISTBase):
    def __init__(self, split="letters", **kwargs):
        super().__init__(datadir="EMNIST_Train", split=split,
                         train=True, download=True, **kwargs)


class EMNISTVal(EMNISTBase):
    def __init__(self, split="letters", **kwargs):
        super().__init__(datadir="EMNIST_Val", split=split,
                         train=False, download=True, **kwargs)
