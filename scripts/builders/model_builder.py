import torchvision.models as models
import os
from MobileNetV1.MobileNetv1 import mobilenet_w1, get_mobilenet


__AVAI_MODELS__ = {
                    'mobilenet_v3_large', 'mobilenet_v1', 'mobilenet_v2',
                  }

def build_model(model_name):
    assert model_name in __AVAI_MODELS__, f"Wrong model name parameter. Expected one of {__AVAI_MODELS__}"

    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained = True)


    if model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained = True)

    if model_name == 'mobilenet_v1':
        root = os.path.join(os.getenv('MODELS_ROOT'), ".torch", "models") if os.getenv('MODELS_ROOT') else \
            os.path.join("~", ".torch", "models")
        model = get_mobilenet(width_scale=1.0, model_name="mobilenet_w1", root = root, pretrained = True)

    return model
