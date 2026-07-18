from setuptools import setup

setup(
        name="qwenvl-embedding",
        version="0.1",
        packages=["embedding"],
        python_requires=">=3.11",
        install_requires=[
            # model + inference
            'torch==2.8.*',
            # torchvision must stay importable by qwen-vl-utils and track torch: 
            # 0.23.x pairs with torch 2.8 but 0.24 needs torch 2.9 and 0.28 needs torch 2.13.
            # So cap torchvision below 0.24.
            'torchvision>=0.23.0,<0.24',
            'transformers>=4.57.3',
            'accelerate>=1.12.0',
            'qwen-vl-utils>=0.0.14',
            # Video reader for qwen-vl-utils (preference order torchcodec > decord > torchvision).
            # decord is used because it is torch-agnostic and has wheels for the container's Python 3.11. 
            # torchcodec's PyPI wheels are pinned to a CUDA 13 build and fail to load on this cu128/torch-2.8 stack.
            # torchvision's reader is deprecated (removed in 0.24) and only a last-resort fallback.
            'decord>=0.6.0',
            'opencv-python-headless>=4.12.0.88',
            'Pillow>=10.0.0',
            'numpy',
            # video I/O used by common_ml.video_processing (PyAV + ffmpeg CLI)
            'av',
            # tagger runtime / plumbing
            'loguru==0.5.2',
            'setproctitle',
            'dacite',
            'ujson',
            'tqdm',
            'pyyaml',
            'common-ml @ git+https://github.com/eluv-io/common-ml@vector-tags',
        ],
        extras_require={
            'test': ['pytest'],
        },
)
