FROM public.ecr.aws/lambda/python:3.12

# Set HuggingFace cache directory to /tmp for Lambda compatibility
ENV HF_HOME=/tmp/hf_home

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install -r requirements.txt && rm -frv /root/.cache

RUN pip install --upgrade torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && rm -frv /root/.cache

COPY load_model.py ./
RUN python ./load_model.py


# Copy function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "lambda_function.handler" ]