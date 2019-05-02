Based on the [Modular semantic segmentation](https://github.com/ethz-asl/modular_semantic_segmentation) framework from the ETHZ ASL repository.
Corresponding implementations for the IROS 2018 paper ["Modular Sensor Fusion for Semantic Segmentation"](https://arxiv.org/abs/1807.11249) by Hermann Blum, Abel Gawel, Roland Siegwart and Cesar Cadena.

# Introduction
In this project the idea of quantifying the uncertainty of semantic segmentation networks is explored. This is achieved by utilizing the generative adversarial networks (GAN) ability to generate synthetic images. In this case, a conditional GAN (cGAN) is trained to learn a transformation from semantic segmentations back to RGB images. This can be useful, when the semantic segmentation fails to recognize or misclassifies foreign (Out-of-Distribution) objects. The cGAN then generates a synthetic image based on the semantic segmentation output.

# Semantic Segmentation
To have a base segmantation network, the framework already included an AdapNet implementation that could easily be trained on the Cityscapes dataset. The only modification necessary, was the use of a different aspect ratio (1:1, instead of 2:1) and size of image to work with the conditional GAN.

<table>
  <tr>
    <td>
       <img height="400px" src="https://github.com/davesean/modular_semantic_segmentation/blob/publish/images/target_1.png">
    </td>
    <td>
       <img height="400px" src="https://github.com/davesean/modular_semantic_segmentation/blob/publish/images/input_1.png">
    </td>
  </tr>
  <tr>
      <td>
          <a> Urban Image with Unknown Objects<sup>1</sup> </a>
      </td>
      <td>
          <a> Corresponing Semantic Segmentation</a>
      </td>
  </tr>
</table>

<sup>1</sup>Images taken from semester thesis "Uncertainties for Deep Learning-based Classification" by Sarlin. (2018)

# Conditional GAN
The cGAN in this implementation was based off the paper ["Image-to-Image Translation with Conditional Adversarial Networks"](https://arxiv.org/abs/1611.07004) by Isola et al (2017). As this implementation uses PyTorch, two implementation of this paper in Tensorflow were combined. ([Affinelayer's](https://github.com/affinelayer/pix2pix-tensorflow) and [Yenchenlin's](https://github.com/yenchenlin/pix2pix-tensorflow)) The cGAN is also trained on the Cityscapes dataset.

<table>
  <tr>
    <td>
       <img height="400px" src="https://github.com/davesean/modular_semantic_segmentation/blob/publish/images/input_1.png">
    </td>
    <td>
       <img height="400px" src="https://github.com/davesean/modular_semantic_segmentation/blob/publish/images/synth_1.png">
    </td>
  </tr>
  <tr>
      <td>
          <a> Semantic Segmentation </a>
      </td>
      <td>
          <a> Generated Synthetic Image</a>
      </td>
  </tr>
</table>

# Patch Discriminator
To quantify uncertainty, a network is needed that detects similarity or rather dissimilarity. Additionally, this dissimilarity needs to be detected locally. Therefore, a patch discriminator is trained on the Cityscapes RGB images and on the generated synthetic images. The patch discriminator is given triplets of patches, where the first patch comes from the RGB image. The second patch is taken from corresponding location in the synthetic image. The third patch is randomly cropped out of another synthetic image. In training, the first two patches are passed to the patch discriminator and labelled as a positive example, meaning that the network should return a 1. Then the network receives the first and third image as a negative example, expecting the network to return a 0.

# Training cGAN
First you'll need to train your conditional GAN. In the config file, set type to train and also how many epochs you want to train. You can also add a checkpoint, to continue training.
```
python -m experiments.run_cGAN with configs/config_pix2pix.yaml --name="cGAN_train
```
The training of the segmentation network can be found in the original modular_semantic_segmentation framework.

# Generating the synthetic dataset
To generate the synthetic images for the complete dataset, you'll need to run with mode set to 'transf'.
This can take a while and requires some memory. You'll need to set the checkpoint of the previously trained cGAN.
It is important that all augmentations are **not** applied.
```
python -m experiments.run_cGAN with configs/config_pix2pix.yaml net_config.type='Unet' net_config.mode='transf' net_config.checkpoint=327 net_config.file_output_dir="/cluster/work/riner/users/haldavid/GeneratedSet" dataset.resize=False dataset.augmentation.vflip=False dataset.augmentation.brightness=False dataset.augmentation.contrast=False dataset.augmentation.gamma=False net_config.use_grayscale=True --name="cGAN_200ep_bw_transf"
```

# Training a dissimilarity detector
Now that you have a training set with real images, synthetic images and semantic segmentations, you can train the dissimilarity detector. The following script trains the dissimilarity detector and directly evaluates on another dataset for you. The dissimilarity detector is trained solely on patches of the real and synthetic images of the training set.
```
python -m experiments.train_and_eval_simDisc with configs/config_pipeline.yaml gan_config.checkpoint=347 --name="arch4_ppd8_347"
```
You can add more architectures to 'xview/models/similarityArchitectures'. Define them like any of the other archs, and don't forget to **add it to the dict** after you define the function.
