########## PyTorch 1.4 Dev #################

FROM twosixarmory/pytorch:0.9.1

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends build-essential gcc libsndfile1 

RUN /opt/conda/bin/pip install \
    torchaudio==0.4.0 \
    librosa==0.7.2 \
    llvmlite==0.32.1 \
    numba==0.43.0 \
    scipy==1.4.1  --no-cache-dir

#fix: numba -- 0.43.0 cache permission issue
#btw, we are using older numba because of librosa
CMD ["mkdir", "/tmp/numba_cache"]
CMD ["chmod", "777", "/tmp/numba_cache"]
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

#install ajati/armory -- scripts needed to incorporate Lipschitz regularization
RUN git clone https://github.com/ajati/armory.git
RUN /opt/conda/bin/pip install ./armory/ --no-cache-dir

CMD tail -f /dev/null
