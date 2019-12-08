DAVIS_path='./datasets/DAVIS/'
save_path='./datasets/DAVIS/OpticalFlow/'
mkdir -p $save_path'Flownet2'

flow_folder_dir=$DAVIS_path"JPEGImages/480p/*/"

for d in $flow_folder_dir
 do
     load_dir=$DAVIS_path'JPEGImages/480p/'`basename $d`'/'
	 save_dir=$save_path'Flownet2/'`basename $d`
	 mkdir -p $save_dir
	 
	 num_frames=`find . -maxdepth 1 -type f -printf . | wc -c`
	 
	 python3 main.py --inference --model FlowNet2 --save_flow \
	 --inference_dataset ImagesFromFolder \
	 --inference_dataset_root $load_dir \
	 --resume checkpoints/FlowNet2_checkpoint.pth.tar \
	 --save $save_dir
	 
	 mv $save_dir"/inference/run.epoch-0-flow-field"/* $save_dir"/"
	 rm -rf $save_dir'/validation'
	 rm -rf $save_dir'/inference'
	 rm -rf $save_dir'/train'
	 rm -rf $save_dir'/args.txt'
done
