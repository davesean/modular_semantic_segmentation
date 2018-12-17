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

# Evaluation
