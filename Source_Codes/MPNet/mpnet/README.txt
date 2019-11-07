1. Introduction

  This is a public implementation of the method described in the following paper: Learning Motion Patterns in Videos [CVPR 2017] (https://hal.inria.fr/
hal-01427480/document). Please report any issues to Pavel Tokmakov (Pavel.Tokmakov@inria.fr).


2. Installation

  Our implementation is based on the Torch framework (http://torch.ch). In addition you will need to install the SharpMask object proposals generation method, also based on Torch (https://github.com/facebookresearch/deepmask), and to have a relatively recently version of Matlab available.


3. Preprocessing
 
  To train the model yourself you will need the FlyingThings3D (FT3D) dataset (https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) with the additional moving object labels available on our project web page. After downloading and unpacking the dataset and the labels, open Matlab in the project's 'matlab' subdirectory, change the path to FT3D in the 'convertFlowFT3D.m' script and run the following command to convert the optical flow to appropriate fromat:

	convertFlowFT3D('TRAIN')

if you plan to evaluate on FT3D, run the following command as well:

	convertFlowFT3D('TEST')

  To evaluate on the DAVIS dataset (http://davischallenge.org/code.html) you need to download and unpack the original 2016 version of the dataset as well as the evaluation code. To estimate the optical flow for this dataset with the GPU implentation of LDOF algorith, open Matlab in the 'matlab' subdirectory, modify the path to DAVIS in the 'computeFlowDAVIS.m' script and execute it. Finally, you will need to estimate the objectness masks for the DAVIS videos. After installing the SharpMask, copy the compute_objectness_DAVIS.lua script into the ShapMask directory, modify the path to DAVIS in the script and execute the following command:

	th estimateDAVIS.lua pretrained/sharpmask -np 100


4. Training

  To train the model, modify the path to the FT3D dataset in the form_batch_angle.lua script and execute the following command:

	th train.lua -gpu $GPU_ID

  where GPU_ID stands for the index of the gpu on which you wish to run the training, starting from 0. The trained model is saved to the 'models' folder under the name 'model.dat'


5. Evaluation

  To evaluate a model on the FT3D dataset, modify the path to the dataset in the 'test_FT3D.lua' script and run:

	th test_FT3D.lua -gpu $GPU_ID -model $MODEL_NAME
  
  where GPU_ID stands for the index of the gpu on which you wish to run the training, starting from 0, and MODEL_NAME stands for the name of the model file you wish to evaluate.

  To evaluate on DAVIs, modify the path in the 'test_DAVIS.lua' script and run:

	th test_DAVIS.lua -gpu $GPU_ID -model $MODEL_NAME -setting $SETTING

  where GPU_ID and MODEL_NAME are as above and SETTING stands for the name of the folder in which the resulting segmentations will be saved, under the $DAVIS/Results/Segmentations/480p directory. Not that this script does not include the CRF postprocessing step. To evaluate the perfromance with the 3 standards DAVIS measures please use the Matlab evaluation code provided with the dataset. 

  When evaluating on other datasets the inputs have to be resize so that the smallest dimension equals to 232. We provide 2 pretrained models, one which was used to produced the results reported in the paper (model_paper.dat) and an updated version (model_new.dat). The difference is that in the new model the upsampling layers resize the input feature maps to exactly the same dimensions instead of simply upsampling the smaller one by a factor of 2, which simplifies handling of the inputs of different spatial resolutions. Oterwise the models are identical and produce very close results. We recommned using the new model on other datasets.


6. CRF postprocessing

  To run the CRF postprocessing step, specify the path to DAVIS dataset in the 'applyCRF.py' script run the following command:

	python applyCRF.py $SETTING $SETTING_CRF

  where SETTING stands for the name of the folder containing the segmentation produced by the model and SETTING_CRF is the name of the folder in which the postprocessed segmentations will be saved.


7. License

We use the BSD 3-clause license. Check the file license.txt for more details.


8. References

If you use this software for research purposes, you should cite the following papers in any resulting publication.

   "Learning motion patterns in videos"
   P. Tokmakov. K. Alahari, C. Schmid
   In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

   "A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation"
   N.Mayer and E.Ilg and P.Haeusser and P.Fischer and D.Cremers and A.Dosovitskiy and T.Brox
   In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

   "Learning to Refine Object Segments"
   P. O. Pinheiro and T. Lin and R. Collobert and P. Dollar
   In European Conference on Computer Vision (ECCV), 2016.

   "Large Displacement Optical Flow: descriptor matching in variational motion estimation"
   T. Brox, J. Malik
   In IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(3): 500-513, 2011.

   "Dense point trajectories by GPU-accelerated large displacement optical flow"
   N. Sundaram, T. Brox, K. Keutzer
   In European Conference on Computer Vision (ECCV), 2010.

