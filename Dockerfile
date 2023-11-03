FROM pytorch/pytorch 
#TODO(auberon) pin to specific image version?

# TODO(auberon) Maintain separate docker images for CPU/GPU versions
COPY ./cpu_requirements.txt /requirements/requirements.txt
RUN pip install -r /requirements/requirements.txt

RUN mkdir /output

COPY ./models /models
COPY ./collage /collage
COPY ./generate.py /generate.py

# Expect user to mount input fasta to /input/input.fasta
# TODO(auberon): Docker-specific error message if they don't?

# TODO(auberon): Allow user to set model easily
ENTRYPOINT ["python", "/generate.py", "/input/input.fasta", "/models/Ecoli.pt", "/output/preds"]