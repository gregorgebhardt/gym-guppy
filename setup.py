from setuptools import setup

setup(name='gym_guppy',
      version='0.0.1',
      install_requires=['gym', 'box2d-py', 'numpy', 'scipy', 'pygame', 'matplotlib', 'imageio', 'imageio-ffmpeg',
                        'numba', 'ray[rllib,tune]']
)