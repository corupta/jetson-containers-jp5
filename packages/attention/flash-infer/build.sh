#!/usr/bin/env bash
set -ex

echo "Building FlashInfer ${FLASHINFER_VERSION}"

REPO_URL="https://github.com/flashinfer-ai/flashinfer"
REPO_DIR="/opt/flashinfer"

git clone $REPO_URL $REPO_DIR
cd $REPO_DIR
git checkout v${FLASHINFER_VERSION_SPEC}
git submodule update --init --recursive
git apply /tmp/flashinfer/patch.diff
sed -i 's|options={.*| |g' setup.py
echo "Patched $REPO_DIR/setup.py"
cat setup.py
sed -i 's/^license = "Apache-2.0"/license = { text = "Apache License 2.0" }/' \
  pyproject.toml

python3 -m flashinfer.aot --f8-dtype
python -m pip install --no-build-isolation --verbose .

# wheel build cmd errs, can be prolly fixed if tinkered with, but I don't upload to twine anyways.
# python3 setup.py --verbose bdist_wheel --dist-dir $PIP_WHEEL_DIR 
# pip3 install $PIP_WHEEL_DIR/flashinfer*.whl

# twine upload --verbose $PIP_WHEEL_DIR/flashinfer*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
