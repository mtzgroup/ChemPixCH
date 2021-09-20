FROM continuumio/miniconda:latest

WORKDIR /code

RUN wget -O chempix_model_params.tar.gz https://osf.io/eh6x4/download && \
    tar -xzf chempix_model_params.tar.gz && \
    rm chempix_model_params.tar.gz

COPY im2smiles/environment.yml .

RUN conda env create -f environment.yml && \
    echo "source activate chempix" > ~/.bashrc
ENV PATH /opt/conda/envs/chempix/bin:$PATH

COPY im2smiles/ .

EXPOSE 8000

CMD ["sh", "-c", "gunicorn server:app -w 2 -k uvicorn.workers.UvicornWorker --keep-alive 650 -b 0.0.0.0:8000 --timeout 0 --log-level debug --access-logfile -"]
