FROM pytorch/pytorch 
#TODO(auberon) pin to specific image version?

# TODO(auberon) Maintain separate docker images for CPU/GPU versions
COPY ./cpu_requirements.txt /requirements/requirements.txt
RUN pip install -r /requirements/requirements.txt

RUN mkdir /output

ENV PYTHONPATH "${PYTHONPATH}:/src/"

COPY ./models /models
COPY ./collage /src/collage
COPY ./generate.py /scripts/generate.py

# Expect user to mount input fasta to /input/input.fasta
# TODO(auberon): Docker-specific error message if they don't?

# TODO(auberon): Allow user to set model easily
ENTRYPOINT ["python", "/scripts/generate.py", "/input/input.fasta", "/models/Ecoli.pt", "/output/preds"]