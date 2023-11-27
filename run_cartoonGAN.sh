relative_path="cartoonGAN/model_ckpt/"
directory_path="$(pwd)/$relative_path"

if [ -d "$directory_path" ]; then
  echo "Cartoonize Started."
else
  sh ./cartoonGAN/download_model.sh
fi

python ./cartoonGAN/generate.py
