import os
import faiss
import numpy as np
from tqdm import tqdm

class VectorDatabaseModel:
    def __init__(self,feature_model,seg_model,database_path):
        self.feature_model=feature_model
        self.seg_model=seg_model
        self.database_path=database_path
        self.faiss_model = faiss.IndexFlatL2(1280)
        self.faiss_model.metric_type = 0
        self.clothes_dic = {}
        self.len=0
        self.classes_num={}
        self.process_database()
    def process_database(self):
        print('processing Database images...')

        for img in tqdm(os.listdir(self.database_path)):
            masks, classes,orig_img = self.seg_model.predict('{}\\{}'.format(self.database_path,img))
            for m, cls in zip(masks, classes):
                features=self.feature_model.extract_features(m)
                self.faiss_model.add(np.asarray(features))
                self.clothes_dic[self.len]=['{}\\{}'.format(self.database_path,img),cls]
                self.len += 1
                try:
                    self.classes_num[cls]+=1
                except:
                    self.classes_num[cls]=1
        print('Number of items: {}'.format(self.classes_num))

    def get_simillers(self,img,clas,k=2):
        features = self.feature_model.extract_features(img)
        dist, items = self.faiss_model.search(np.asarray(features, dtype=np.float32),k)
        images = [self.clothes_dic[im_id] for im_id in items[0] if self.clothes_dic[im_id][1]==clas]
        return images