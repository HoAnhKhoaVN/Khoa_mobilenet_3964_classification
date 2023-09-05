from torch.nn import Module, Linear
from torchvision.models import EfficientNet_B3_Weights
from torchvision.models import efficientnet_b3

class MyEfficientnetB3(Module):
    def __init__(
        self,
        num_classes: int = 4
    ):
        super().__init__()
        self.base_model = efficientnet_b3(weights = EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = Linear(in_features, num_classes)
        
    def forward(self, x):
        x = self.base_model(x)
        return x