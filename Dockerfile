
FROM continuumio/miniconda3

COPY sample-app/environment.yml /tmp/environment.yml
WORKDIR /tmp
RUN apt-get update && \
    apt-get install -y --no-install-recommends libxrender1 libsm6 && \
    conda update -n base -c defaults conda && \
    conda env update -f environment.yml && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    conda clean --all --yes

RUN mkdir -p /deploy/app
COPY sample-app /deploy/app

WORKDIR /deploy/app

CMD gunicorn --bind 0.0.0.0:$PORT main:app