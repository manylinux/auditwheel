#!/bin/bash

# This shall be run on manylinux2014_x86_64 to create test data for gepetto_example_adder

# Stop at any error, show all commands
set -exuo pipefail

OUTPUT_DIR=$(dirname $0)

#mkdir /root/maketest
cd /root/maketest

# Install ninja
PY38_BIN=/opt/python/cp38-cp38/bin
$PY38_BIN/pip install ninja
ln -sf $PY38_BIN/ninja /usr/local/bin/
ln -sf $PY38_BIN/wheel /usr/local/bin/

# build boost
curl -fsSLO https://dl.bintray.com/boostorg/release/1.72.0/source/boost_1_72_0.tar.bz2
tar -xf boost_1_72_0.tar.bz2
pushd boost_1_72_0
./bootstrap.sh --prefix=/usr/local
sed -i 's/using python.*/python_config/g' project-config.jam
sed -i 's&python_config&python_config\n    using python : 3.9 : /opt/python/cp39-cp39/bin/python : /opt/python/cp39-cp39/include/python3.9 : /opt/python/cp39-cp39/lib ;&g' project-config.jam
sed -i 's&python_config&python_config\n    using python : 3.8 : /opt/python/cp38-cp38/bin/python : /opt/python/cp38-cp38/include/python3.8 : /opt/python/cp38-cp38/lib ;&g' project-config.jam
sed -i 's&python_config&python_config\n    using python : 3.7 : /opt/python/cp37-cp37m/bin/python : /opt/python/cp37-cp37m/include/python3.7m : /opt/python/cp37-cp37m/lib ;&g' project-config.jam
sed -i 's&python_config&python_config\n    using python : 3.6 : /opt/python/cp36-cp36m/bin/python : /opt/python/cp36-cp36m/include/python3.6m : /opt/python/cp36-cp36m/lib ;&g' project-config.jam
sed -i 's/python_config//g' project-config.jam

./b2 install link=shared python=3.6,3.7,3.8,3.9 --with-python --with-test -j"$(nproc)"
popd

# build example-adder
git clone --recursive https://github.com/Ozon2/example-adder.git
pushd example-adder
git checkout d319dae3849b9dc3161b2b6cbafa9e45204dcc14

for PYBIN in /opt/python/{cp36*,cp37*,cp38*,cp39*}/bin; do
	rm -rf _skbuild/
	"$PYBIN"/pip install --upgrade pip
	"$PYBIN"/pip install scikit-build
	"$PYBIN"/python setup.py bdist_wheel
done
popd

# strip/compress dependencies
for PYVER in 36 37 38 39; do
	strip --strip-unneeded /usr/local/lib/libboost_python${PYVER}.so.1.72.0
	xz -z -c -e /usr/local/lib/libboost_python${PYVER}.so.1.72.0 > ${OUTPUT_DIR}/libboost_python${PYVER}.so.1.72.0.xz
done
strip --strip-unneeded example-adder/_skbuild/linux-x86_64-3.9/cmake-build/libexample-adder.so.3.0.2-6-gd319
xz -z -c -e example-adder/_skbuild/linux-x86_64-3.9/cmake-build/libexample-adder.so.3.0.2-6-gd319 > ${OUTPUT_DIR}/libexample-adder.so.3.0.2-6-gd319.xz

# copy wheels
cp example-adder/dist/* ${OUTPUT_DIR}/
