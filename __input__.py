import numpy as np
import os
import fastai
from fastai.vision import *
from mask_functions import *
from sklearn.model_selection import KFold

def data(fold):
  sz = 256
  bs = 16
  n_acc = 64//bs #gradinet accumulation steps
  nfolds = 4
  SEED = 2019
  #SEEDING
  random.seed(SEED)
  os.environ['PYTHONHASHSEED'] = str(SEED)
  np.random.seed(SEED)
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)
  torch.backends.cudnn.deterministic = True

  #eliminate all predictions with a few (noise_th) pixesls
  noise_th = 75.0*(sz/128.0)**2 #threshold for the number of predicted pixels
  best_thr0 = 0.2 #preliminary value of the threshold for metric calculation

  if sz == 256:
      stats = ([0.540,0.540,0.540],[0.264,0.264,0.264])
      TRAIN = '../input/siimacr-pneumothorax-segmentation-data-256/train'
      TEST = '../input/siimacr-pneumothorax-segmentation-data-256/test'
      MASKS = '../input/siimacr-pneumothorax-segmentation-data-256/masks'
  elif sz == 128:
      stats = ([0.615,0.615,0.615],[0.291,0.291,0.291])
      TRAIN = '../input/siimacr-pneumothorax-segmentation-data-128/train'
      TEST = '../input/siimacr-pneumothorax-segmentation-data-128/test'
      MASKS = '../input/siimacr-pneumothorax-segmentation-data-128/masks'

  # copy pretrained weights for resnet34 to the folder fastai will search by default
  Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)
  kf = KFold(n_splits=nfolds, shuffle=True, random_state=SEED)
  valid_idx = list(kf.split(list(range(len(Path(TRAIN).ls())))))[fold][1]
  # Create databunch
  data = (SegmentationItemList.from_folder(TRAIN)
          .split_by_idx(valid_idx)
          .label_from_func(lambda x : str(x).replace('train', 'masks'), classes=[0,1])
          .add_test(Path(TEST).ls(), label=None)
          .transform(get_transforms(), size=sz, tfm_y=True)
          .databunch(path=Path('.'), bs=bs)
          .normalize(stats))
  return data
