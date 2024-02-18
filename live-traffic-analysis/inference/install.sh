# new venv

python3 -m venv venv
source ./venv/bin/activate
python -m pip install --upgrade pip
pip install keyring artifacts-keyring
pip install coloredlogs flatbuffers numpy packaging protobuf sympy ffmpeg-python opencv-python-headless torch_tps psycopg2-binary redis supervision inference python-dotenv scikit-learn pyarrow
pip install onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
