import os
from PIL import Image
from torchvision import models, transforms
import os
import torch
import json
import time
import traceback
from threading import Timer
from torch.autograd import Variable
from abstractApi import AbstractModel
# from .dm import DM_Module
from .conf import model_param
# from .log import logger
# from .ruler import RegexRulePool
# from .corpus_db import CorpusDB
# from multiprocessing import Pool
import asyncio
import threading

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class FeatureDerive(AbstractModel):
    """FeatureDerive class."""

    def __init__(self):
        pass

    # def loadParam(self, fparam):
    #     """Load parameters."""
    #     self.ruler_pool = RegexRulePool()
    #     self.corpus_db = CorpusDB()
    #     self.dm = DM_Module()
    #     #
    #     # self.timer = Timer(300, self.reload_param)
    #     # self.timer.setDaemon(True)
    #     # self.timer.start()

    def pil_loader(imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
    # def reload_param(self):
    #     """Reload all parameters every 10 minutes."""
    #     self.ruler_pool = RegexRulePool()
    #     self.corpus_db = CorpusDB()
    #     self.dm = DM_Module()
    #
    #     # self.timer = Timer(300, self.reload_param)
    #     # self.timer.setDaemon(True)
    #     # self.timer.start()

    def predict(self, param):
        resume = ''
        test_transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224, scale=(0.49, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = self.pil_loader(param)
        if transforms is not None:
            img = test_transforms(img)
        inputs = Variable(img.cuda())
        model = models.resnet18()
        model.fc = torch.nn.Linear(512, 2)
        model.load_state_dict(torch.load(resume))
        model = model.cuda()
        model.eval()
        outputs = model(inputs)
        if isinstance(outputs, list) or isinstance(outputs, tuple):
            outputs = (outputs[0] + outputs[1]) / 2
        _, preds = torch.max(outputs, 1)
        result = preds.cpu().numpy()[0]
        return result



if __name__ == '__main__':
    # from conf import model_param
    a = FeatureDerive()
    # a.loadParam(model_param['fileParam'])
    param = ''
    print('a.predict ', a.predict(param))


