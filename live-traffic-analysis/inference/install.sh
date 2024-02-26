# new venv

python3 -m venv venv
source ./venv/bin/activate
python -m pip install --upgrade pip
pip install keyring artifacts-keyring
pip install coloredlogs flatbuffers numpy packaging protobuf sympy ffmpeg-python opencv-python-headless torch_tps psycopg2-binary redis supervision inference python-dotenv scikit-learn pyarrow ultralytics
pip install onnxruntime-gpu --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/


cp venv/lib/python3.10/site-packages/supervision/detection/utils.py ./overloaded_libraries/detections/
mv venv/lib/python3.10/site-packages/supervision/detection/utils.py venv/lib/python3.10/site-packages/supervision/detection/original_utils.py
cd venv/lib/python3.10/site-packages/supervision/detection
ln -s ../../../../../../overloaded_libraries/detections/utils.py .
cd ../../../../../..
