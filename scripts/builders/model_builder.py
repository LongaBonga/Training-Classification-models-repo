import torchvision.models as models
from MobileNetV1.MobileNetv1 import mobilenet_w1


__AVAI_MODELS__ = {
                    'mobilenet_v3_large', 'mobilenet_v1', 'mobilenet_v2',
                  }

def build_model(model_name):
    assert model_name in __AVAI_MODELS__, f"Wrong model name parameter. Expected one of {__AVAI_MODELS__}"

    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained = True)


    if model_name == 'mobilenet_v3_large':
        model = models.mobilenetv3_large(pretrained = True)

    if model_name == 'mobilenet_v1':
        model = mobilenet_w1(pretrained = True)

    return model
