
def data(sz):
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
      
  return (TRAIN,TEST)
