from builders.model_builder import load_pretrained_weights
from subprocess import run, DEVNULL, CalledProcessError
def export_onnx(model, snapshot_path, img_size=(128,128), save_path='model.onnx'):
    # input to inference model
    dummy_input = torch.rand(size=(1,3,*img_size))
    dummy_cat = torch.zeros(1, dtype=torch.long)
    # load checkpoint from config
    load_pretrained_weights(model, snapshot_path)
    # convert model to onnx
    input_names = ["data"]
    output_names = ["cls_bbox"]
    with torch.no_grad():
        model.eval()
        torch.onnx.export(model, args=dummy_input, f=save_path, verbose=True,
                      input_names=input_names, output_names=output_names)

def export_mo(onnx_model_path, mean_values, scale_values, save_path):
    command_line = (f'mo.py --input_model="{onnx_model_path}" '
                   f'--mean_values="{mean_values}" '
                   f'--scale_values="{scale_values}" '
                   f'--output_dir="{save_path}" '
                   f'--reverse_input_channels ')

    try:
        run('mo.py -h', stdout=DEVNULL, stderr=DEVNULL, shell=True, check=True)
    except CalledProcessError as _:
        print('OpenVINO Model Optimizer not found, please source '
            'openvino/bin/setupvars.sh before running this script.')
        return

    run(command_line, shell=True, check=True)