from dacite import from_dict
import setproctitle

from common_ml.tagging.run_helpers import catch_errors, get_params, run_default

from siglip_frame.model import FeatureExtractor
from siglip_frame.config import RuntimeConfig
from config import config

if __name__ == '__main__':
    setproctitle.setproctitle('siglip2-frame-vectors')

    catch_errors()

    params = get_params()

    params = from_dict(RuntimeConfig, data=params)

    # One FrameVector per sampled frame
    model = FeatureExtractor(
        cfg=params,
        model_id=config["model"]["model_id"],
        revision=config["model"].get("revision"),
    )

    run_default(model)
