
from jetson_containers import CUDA_VERSION
from packaging.version import Version
import os

# todo refactor here maybe :)
if CUDA_VERSION <= Version('12.6') or os.environ.get('NUMPY_VERSION', '2').startswith('1'):
  NUMPY_PACKAGE = 'numpy<2'
  NUMPY_VERSION_MAJOR = 1
else:
  NUMPY_PACKAGE = 'numpy'
  NUMPY_VERSION_MAJOR = 2

package['build_args'] = {
  'NUMPY_PACKAGE': NUMPY_PACKAGE,
  'NUMPY_VERSION_MAJOR': NUMPY_VERSION_MAJOR,
}