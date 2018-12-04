import os
import glob
import sacred as sc
import cv2
import scipy.misc
import numpy as np
import tensorflow as tf
from sacred.utils import apply_backspaces_and_linefeeds
from experiments.utils import get_observer, load_data
from xview.datasets import Cityscapes
from experiments.evaluation import evaluate, import_weights_into_network
from xview.datasets import get_dataset
from xview.models import get_model
from xview.settings import EXP_OUT
import sys
sys.path.insert(0, '../pix2pix-tensorflow')
from model import pix2pix
import shutil

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

def predict_network(net, output_dir, image_path, data_desc):
    """
    Predict on a given dataset.

    Args:
        net: An instance of a `base_model` class.
        output_dir: A directory path. This function will add all files found at this path
            as artifacts to the experiment.
        image_path: Path to the image
    """
    seg = []
    inp = []
    img_w=256
    img_h=256

    try:
        image = np.float32(np.expand_dims(cv2.resize(cv2.imread(image_path),(img_w,img_h),interpolation=cv2.INTER_LINEAR), axis=0))
        inp.append(image)
        data = {'rgb': image, 'depth': tf.zeros(shape=[1, img_w, img_h, 1],dtype=tf.float32), 'labels': tf.zeros(shape=[1, img_w, img_h],dtype=tf.int32)}
        output = net.predict(data)
        outputColor = data_desc.coloured_labels(labels=output)
        seg.append(outputColor)

    except KeyboardInterrupt:
        print('WARNING: Got Keyboard Interrupt')

    # return np.concatenate(ret)
    return seg, inp

    # To end the experiment, we collect all produced output files and store them.
    # for filename in os.listdir(output_dir):
    #     experiment.add_artifact(os.path.join(output_dir, filename))

def load_list_path(input_path):
    training = glob.glob(os.path.join(input_path,"training","target_training*.png"))
    training.sort()
    measure = glob.glob(os.path.join(input_path,"measure","target_measure*.png"))
    measure.sort()
    validation = glob.glob(os.path.join(input_path,"validation","target_validation*.png"))
    validation.sort()
    tmp = {'training': training, 'measure': measure, 'validation': validation}
    return tmp

ex = sc.Experiment()
# reduce output of progress bars
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(get_observer())

@ex.capture
def predict_output(net, output_dir, image_path, data_desc, _run):
    """Predict data on a given network"""
    return predict_network(net, output_dir, image_path, data_desc)


@ex.main
def main(modelname, net_config, gan_config, dataset, datasetGAN, starting_weights, input_folder, _run):
    for key in gan_config:
        setattr(a, key, gan_config[key])

    setattr(a,'EXP_OUT',EXP_OUT)
    setattr(a,'RUN_id',_run._id)

    # Set up the directories for diagnostics
    output_dir = create_directories(_run._id, ex)

    # load the data for the data description
    data_desc = get_dataset(dataset['name'])
    data_desc = data_desc()
    # load images in input_folder
    path_dicts = load_list_path(input_folder)

    # create the network
    model = get_model(modelname)
    net = model(data_description=data_desc.get_data_description(),output_dir=output_dir, **net_config)
    net.import_weights(filepath=starting_weights)
    print("INFO: semSegNet Imported weights succesfully")

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    data = get_dataset(datasetGAN['name'])
    # data = data(**dataset)

    model = pix2pix(sess, image_size=a.input_image_size, batch_size=a.batch_size,
                    output_size=a.input_image_size, dataset_name=dataset['name'],
                    checkpoint_dir=output_dir, data_desc=data.get_data_description(), momentum=a.batch_momentum,
                    L1_lambda=int(a.l1_weight/a.gan_weight), gf_dim=a.ngf,
                    df_dim=a.ndf,label_smoothing=a.label_smoothing,
                    noise_std_dev=a.noise_std_dev, checkpoint=os.path.join(a.EXP_OUT,str(a.checkpoint)))
    print("INFO: GAN Imported weights succesfully")

    ss_run_id = starting_weights.split('/')[-2]
    gan_run_id = str(a.checkpoint)
    folder_name = ss_run_id + "_" + gan_run_id

    base_output_path = os.path.join(a.file_output_dir,folder_name)
    print(base_output_path)

    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)

    set_names = ['validation','measure','training']

    for set in set_names:
        set_path = os.path.join(base_output_path,set)

        for image_path in path_dicts[set]:
            num = image_path.split('.')[0].split(set[-1])[-1]
            if not os.path.exists(set_path):
                os.makedirs(set_path)

            sem_seg_images, rgb_images = predict_output(net,output_dir,image_path,data_desc)
            rgb_input, segmentation, synth, realfield, fakefield = model.transform(a,sem_seg_images, rgb_images)

            filename = "target_"+num+".png"
            cv2.imwrite(os.path.join(set_path,filename), rgb_input, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            filename = "segm_"+num+".png"
            cv2.imwrite(os.path.join(set_path,filename), segmentation, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            filename = "synth_"+num+".png"
            cv2.imwrite(os.path.join(set_path,filename), synth, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if set is 'validation':
                filename = "rf_"+num+".png"
                cv2.imwrite(os.path.join(set_path,filename), realfield, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                filename = "ff_"+num+".png"
                cv2.imwrite(os.path.join(set_path,filename), fakefield, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

if __name__ == '__main__':
    ex.run_commandline()
    # for some reason we have processes running in the background that won't stop
    # this is the only way to kill them
    os._exit(os.EX_OK)
