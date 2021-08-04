from .export import export_onnx, export_mo
import numpy as np
import os.path as osp
import os
from openvino.inference_engine import IENetwork, IECore
from help_functions.distributed import print_at_master

def conversion(args, model, snapshot_path, img_size=(128,128), save_path='./'):

    path = osp.join(save_path, f'data_for_OpenVino_inference/{args.model}')
    os.makedirs(path, exist_ok=True)
    save_path = path

    export_onnx(model, snapshot_path, img_size, save_path = save_path + '/model.onnx')

    mean_values = str([0.485, 0.456, 0.406])
    scale_values = str([0.229, 0.224, 0.225])

    path = osp.join(save_path, 'IR/')
    os.makedirs(path, exist_ok=True)
    IR_path = path

    export_mo(save_path + '/model.onnx', mean_values, scale_values, save_path = IR_path)
    print_at_master(f'model conversions successfully! path \n {IR_path}')


def eval_inference(path, inference_data):

    model_xml = '{}/model.xml'.format(path)
    model_bin = '{}/model.bin'.format(path)

    # plugin = IEPlugin("CPU", plugin_dirs=plugin_dir)
    ie_core = IECore()
    # versions = ie.get_versions("CPU")
    # Read IR
    net = net = ie_core.read_network(model=model_xml, weights=model_bin)
    exec_net = ie_core.load_network(network=net, device_name="CPU")
    
    del net

    val_size = len(inference_data)
    val_pred = 0.

    for img, label in inference_data:
        # print(img_numpy)
        pred = exec_net.infer({'data': img})
        val_pred += (pred['clf'].argmax() == label)
    
    return val_pred.item() / val_size
