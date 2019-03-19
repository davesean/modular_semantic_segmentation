import tensorflow as tf
import sys

from experiments.utils import get_observer

from xview.datasets import Cityscapes_generated
from xview.datasets import get_dataset
from xview.models import get_model
from xview.settings import EXP_OUT

import os
import sacred as sc
import shutil
import glob
from sacred.utils import apply_backspaces_and_linefeeds
import tensorflow as tf
import numpy as np

class Helper:
    name = 'A'

a = Helper()

def create_directories(run_id, experiment):
    """
    Make sure directories for storing diagnostics are created and clean.

    Args:
        run_id: ID of the current sacred run, you can get it from _run._id in a captured
            function.
        experiment: The sacred experiment object
    Returns:
        The path to the created output directory you can store your diagnostics to.
    """
    root = EXP_OUT
    # create temporary directory for output files
    if not os.path.exists(root):
        os.makedirs(root)
    # The id of this experiment is stored in the magical _run object we get from the
    # decorator.
    output_dir = '{}/{}'.format(root, run_id)
    if os.path.exists(output_dir):
        # Directory may already exist if run_id is None (in case of an unobserved
        # test-run)
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    # Tell the experiment that this output dir is also used for tensorflow summaries
    experiment.info.setdefault("tensorflow", {}).setdefault("logdirs", [])\
        .append(output_dir)
    return output_dir

ex = sc.Experiment()
# reduce output of progress bars
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_observer())

@ex.main
def main(dataset, net_config, _run):
    # Add all of the config into the helper class
    for key in net_config:
        setattr(a, key, net_config[key])

    setattr(a,'EXP_OUT',EXP_OUT)
    setattr(a,'RUN_id',_run._id)

    output_dir = create_directories(_run._id, ex)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        data = get_dataset(dataset['name'])
        data = data(dataset['image_input_dir'],**dataset)
        data_id=dataset['image_input_dir'].split('/')[-1].split('_')[0]
        setattr(a,'DATA_id',data_id)
        disc_model = get_model('simDisc')
        ckp=None
        if net_config['checkpoint'] is not None:
            ckp = os.path.join(a.EXP_OUT,str(net_config['checkpoint']))
        if net_config['feature_extractor'] is not None:
            fe_ckp = os.path.join(a.EXP_OUT,str(net_config['feature_extractor']))

        model=disc_model(sess=sess, image_size=a.input_image_size,
                     batch_size=a.batch_size, df_dim=a.ndf,
                     input_c_dim=3,
                     checkpoint_dir=output_dir, data=data,
                     momentum=a.batch_momentum, arch=net_config['arch'],
                     checkpoint=ckp, feature_extractor=fe_ckp)
        if a.mode == "train":
            tmp = model.train(a)
            _run.info['predictions'] = tmp
            _run.info['mean_predictions'] = np.mean(tmp, axis=0)
        elif a.mode == "predict":
            input_list = glob.glob(os.path.join(a.predict_dir,"target_*.png"))
            synth_list = glob.glob(os.path.join(a.predict_dir,"synth_*.png"))
            segm_list = glob.glob(os.path.join(a.predict_dir,"input_*.png"))

            input_list.sort(key=lambda x: int(x.partition('_')[-1].partition('.')[0]))
            synth_list.sort(key=lambda x: int(x.partition('_')[-1].partition('.')[0]))
            segm_list.sort(key=lambda x: int(x.partition('_')[-1].partition('.')[0]))

            model.predict(a,input_list,synth_list,segm_list)

if __name__ == '__main__':
    ex.run_commandline()
    # for some reason we have processes running in the background that won't stop
    # this is the only way to kill them
    os._exit(os.EX_OK)
