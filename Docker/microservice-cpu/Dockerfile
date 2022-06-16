#FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
FROM ubuntu:18.04
# Miniconda install copy-pasted from Miniconda's own Dockerfile reachable
# at: https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/debian/Dockerfile

ENV PATH /opt/conda/bin:$PATH

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion curl && \
    apt-get clean

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# Project setup
## install nebula part
RUN  apt-get update \
    && apt-get install -y libgl1-mesa-glx \
    && apt-get install -y ffmpeg libsm6 libxext6 \
    && apt-get install -y nano nodejs \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# # Copy environment.yml (if found) to a temp location so we update the environment. Also
# # copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* /tmp/conda-tmp/
# adding extra channels and creating env
RUN /opt/conda/bin/conda config --append channels conda-forge && /opt/conda/bin/conda config --append channels pytorch && \
    if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && conda clean --all --yes && rm -rf /tmp/conda-tmp

# RUN conda install jupyterlab && conda clean --all --yes
RUN conda clean --all --yes


RUN git clone https://github.com/NEBULA3PR0JECT/nebula3_experts.git && git clone https://github.com/NEBULA3PR0JECT/nebula3_pipeline.git \ 
    && git clone https://github.com/NEBULA3PR0JECT/nebula3_vlm.git && git clone https://github.com/NEBULA3PR0JECT/nebula3_database.git && mkdir notebooks

RUN cp -r /nebula3_experts/. /notebooks/ && cp -r /nebula3_pipeline/. /notebooks/nebula3_pipeline/ \
&& cp -r /nebula3_vlm/. /notebooks/nebula3_vlm/ && cp -r /nebula3_database/. /notebooks/nebula3_pipeline/nebula3_database/ \
&& cp -r /nebula3_database/. /notebooks/nebula3_vlm/nebula3_database/

WORKDIR /notebooks

EXPOSE 8088
CMD ["uvicorn", "test.main:app", "--host", "0.0.0.0", "--port", "8088"]