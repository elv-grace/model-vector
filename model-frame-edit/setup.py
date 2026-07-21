from setuptools import setup

setup(
    name="blip2-frame-vectors",
    version="0.1",
    packages=["blip_frame"],
    python_requires=">=3.11",
    install_requires=[
        # model + inference
        'torch==2.8.*',
        'transformers>=4.57.3',
        'accelerate>=1.12.0',
        'Pillow>=10.0.0',
        'numpy',
        # image read path in common_ml.tagging.file_tagger (images tagged frame-directly)
        'opencv-python-headless>=4.12.0.88',
        # video frame extraction: common_ml.video_processing decodes/samples frames with
        # PyAV when a video file is tagged (the AVModel-from-frame-model path). Also
        # imported unconditionally at startup by common_ml.tagging.run_helpers.
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
