from jetson_containers import CUDA_ARCHITECTURES, CUDA_VERSION, update_dependencies
from packaging.version import Version
   
def nccl(version, branch=None, patch=None, depends=None, requires=None, default=False):
    pkg = package.copy()

    pkg['name'] = f'nccl:{version}'
    
    if not branch:
        branch = 'v' + version
        if len(branch.split('.')) < 3:
            branch = branch + '.0'

    if not patch:
        patch = 'patches/empty.diff'
        
    pkg['build_args'] = {
        'NCCL_VERSION': version,
        'NCCL_BRANCH': branch,
        'NCCL_PATCH': patch,
        'NVCC_GENCODE': ' '.join([f'-gencode=arch=compute_{x},code=sm_{x}' for x in CUDA_ARCHITECTURES]),
    }

    if depends:
        pkg['depends'] = update_dependencies(pkg['depends'], depends)
        
    if requires:
        pkg['requires'] = requires
        
    builder = pkg.copy()
    builder['name'] = builder['name'] + '-builder'
    builder['build_args'] = {**builder['build_args'], 'FORCE_BUILD': 'on'}
    
    if default:
        pkg['alias'] = 'nccl'
        builder['alias'] = 'nccl:builder'
        
    return pkg, builder

package = [
    nccl('2.26.2-1', default=True),
]
