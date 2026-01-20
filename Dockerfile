# See ../triqs/packaging for other options
FROM flatironinstitute/triqs:unstable-ubuntu-clang
ARG APPNAME=triqs_xca

# Install here missing dependencies, e.g.
RUN apt-get update && apt-get install -y \
    clang-19 \
    llvm-19-dev \
    libclang-19-dev \
    libomp-19-dev \
    libzstd-dev \
    doxygen
#    python3-pip # for pip install cvxpy

# Install pyed
RUN git clone https://github.com/HugoStrand/pyed $SRC/pyed
ENV PYTHONPATH=$SRC/pyed:$PYTHONPATH

## Install cvxpy
#RUN pip install --no-cache-dir --break-system-packages cvxpy==1.5.4

# Install adapol
ADD https://api.github.com/repos/flatironinstitute/adapol/git/refs/heads/main adapol_version.json
RUN git clone https://github.com/flatironinstitute/adapol.git $SRC/adapol --branch main
ENV PYTHONPATH=$SRC/adapol/src:$PYTHONPATH

COPY --chown=build . $SRC/$APPNAME
RUN mkdir $BUILD/$APPNAME && chown build $BUILD/$APPNAME

RUN git clone https://github.com/flatironinstitute/clair --branch unstable $SRC/clair && \
    mkdir $BUILD/clair && \
    CXX="clang++-19" CXXFLAGS="" CPLUS_INCLUDE_PATH="" cmake -S $SRC/clair -B $BUILD/clair -DBuild_Tests=OFF -DCMAKE_INSTALL_PREFIX=$INSTALL && \
    cmake --build $BUILD/clair && cmake --install $BUILD/clair

ARG BUILD_ID
ARG CMAKE_ARGS
USER build
WORKDIR $BUILD/$APPNAME
RUN cmake $SRC/$APPNAME -DTRIQS_ROOT=${INSTALL} $CMAKE_ARGS && make -j4 || make -j1 VERBOSE=1
USER root
RUN make install
