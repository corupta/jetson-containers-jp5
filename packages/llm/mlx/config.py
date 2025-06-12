from jetson_containers import L4T_VERSION, update_dependencies

def mlx(commit, patch=None, fork='ml-explore/mlx', version='0.8.0', depends=[], requires=None, default=False):
    pkg = package.copy()
    
    if default:
       pkg['alias'] = 'mlx'
    
    if requires:
        pkg['requires'] = requires

    pkg['name'] = f'mlx:{version}'
    pkg['notes'] = f"[{fork}](https://github.com/{fork}/tree/{commit}) commit SHA [`{commit}`](https://github.com/{fork}/tree/{commit})"
    pkg['depends'] = update_dependencies(pkg['depends'], [*depends])

    pkg['build_args'] = {
        'MLX_VERSION': version,
        'MLX_COMMIT': commit,
        'MLX_FORK': fork,
        'MLX_PATCH': patch
    }

    builder = pkg.copy()

    builder['name'] = f'mlx:{version}-builder'
    builder['build_args'] = {**pkg['build_args'], **{'FORCE_BUILD': 'on'}}

    return pkg, builder

package = [
  mlx('7ecfd673', 'patches/7ecfd673.diff', version='0.26.0', requires='>=35', default=(L4T_VERSION.major >= 35),
    fork='corupta/mlx-cuda', #'frost-beta/mlx-cuda'
    depends=['pytorch:2.7', 'triton:3.2.0', 'torchvision:0.22.0']
  ),
]
