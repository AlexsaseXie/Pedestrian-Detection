import os
from torch.utils.data import Dataset
from src.data_augmentation import *
import pickle
import copy


class NewDataset(Dataset):
    def __init__(self, root_path="data", image_size=416, is_training=True):
        if is_training == True:
            self.image_path = os.path.join(root_path, "train")
            anno_path = os.path.join(root_path, "train_annotations.txt")
            self.read_annotations(anno_path)

        else :
            self.image_path = os.path.join(root_path, "val")
            anno_path = os.path.join(root_path, "val_annotations.txt")
            self.read_annotations(anno_path)

        self.image_size = image_size
        self.num_images = len(self.anno_data)
        self.is_training = is_training

    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        image_path = os.path.join(self.image_path, self.anno_data[item]['image'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        objects = copy.deepcopy(self.anno_data[item]['objects'])

        transformations = Compose([NewResize(self.image_size)])
        image, objects = transformations((image, objects))
        return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), np.array(objects, dtype=np.float32)
    

    def read_annotations(self, anno_path):
        with open(anno_path, 'r') as f:
            anno_list = f.readlines()

        print(len(anno_list))
        self.anno_data = []
        for anno_str in anno_list:
            anno_str = anno_str[:-1]
            tmp_list = anno_str.split(' ')
            tmp = {}
            tmp['image'] = tmp_list[0]
            tmp['objects'] = []

            block = []
            for i in range(1, len(tmp_list)):
                if (i % 5 == 1):
                    continue
                block.append( int(tmp_list[i]) )
                if (i % 5 == 0):
                    block.append(int(tmp_list[i-4]))
                    tmp['objects'].append(block)
                    block = []

            self.anno_data.append(tmp)
