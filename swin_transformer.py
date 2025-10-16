import torchvision.models as models

swin_transformer = models.swin_t(weights=models.Swin_T_Weights.DEFAULT)
