relative_path="SAM/model_ckpt/"
directory_path="$(pwd)/$relative_path"
if [ -d "$directory_path" ]; then
  echo "SAM segmentation Started."
else
  sh ./SAM/download_model.sh
fi

python ./SAM/generate.py --data_dir ./data/cartoonized --save_dir ./data/masks --model_dir ./SAM/model_ckpt/sam_vit_h_4b8939.pth


