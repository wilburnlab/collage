FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

COPY ./tools/requirements/shared_requirements.txt /requirements/requirements.txt
RUN pip install -r /requirements/requirements.txt

RUN mkdir /output

ENV PYTHONPATH "${PYTHONPATH}:/src/"

COPY ./models /models
COPY ./collage /src/collage
COPY ./generate.py /scripts/generate.py

# Expect user to mount input fasta to /input/input.fasta
# TODO(auberon): Docker-specific error message if they don't?

# TODO(auberon): Allow user to set model easily
ENTRYPOINT ["python", "/scripts/generate.py", "/input/input.fasta", "/models/Ecoli.pt", "/output/preds.fasta"]