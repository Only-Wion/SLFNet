param(
    [string]$EnvName = "SLFNet-rtx30"
)

$ErrorActionPreference = "Stop"

conda create -n $EnvName python=3.8 -y
conda install -n $EnvName pytorch==1.10.2 torchvision==0.11.3 cudatoolkit=11.3 -c pytorch -c defaults --override-channels -y
conda install -n $EnvName tqdm matplotlib scikit-image pyyaml -c defaults --override-channels -y
conda run -n $EnvName python -m pip install opencv-python==4.5.5.64 tensorboardX

conda run -n $EnvName python -c "import torch, torchvision, cv2; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('cuda available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'); print('cv2', cv2.__version__)"
