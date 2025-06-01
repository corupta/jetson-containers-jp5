from jetson_containers import L4T_VERSION, update_dependencies

def lmdeploy(commit, patch=None, fork='InternLM', version='0.8.0', depends=[], requires=None, default=False):
    pkg = package.copy()
    
    if default:
       pkg['alias'] = 'lmdeploy'
    
    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'lmdeploy:{version}'
    pkg['notes'] = f"[{fork}/lmdeploy](https://github.com/{fork}/lmdeploy/tree/{commit}) commit SHA [`{commit}`](https://github.com/{fork}/lmdeploy/tree/{commit})"
    pkg['depends'] = update_dependencies(pkg['depends'], [*depends])

    pkg['build_args'] = {
        'LMDEPLOY_VERSION': version,
        'LMDEPLOY_COMMIT': commit,
        'LMDEPLOY_FORK': fork,
        'LMDEPLOY_PATCH': patch
    }

    builder = pkg.copy()

    builder['name'] = f'lmdeploy:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    return pkg, builder

package = [
  lmdeploy('19f6d68', 'patches/19f6d68.diff', version='0.8.1', requires='>=36', default=(L4T_VERSION.major >= 36),
    depends=['pytorch:2.7', 'triton:3.2.0', 'torchvision:0.22.0']
  ),
  # 0.8.2 uses newer gemm to enable blackwell support but also adds fp8 moe support, I'm unsure if it works properly, tho.
  lmdeploy('4a58024f',  'patches/4a58024f.diff',fork='lzhangzz', version='0.8.2', requires='>=36', 
    depends=['pytorch:2.7', 'triton:3.2.0', 'torchvision:0.22.0']
  )
]
