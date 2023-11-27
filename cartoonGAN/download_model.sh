mkdir ./cartoonGAN/model_ckpt

# wget -c http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Hayao_net_G_float.pth
# wget -c http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Hosoda_net_G_float.pth
# wget -c http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Paprika_net_G_float.pth	

# Shinkai style is recommended for clean edge.
wget -c http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Shinkai_net_G_float.pth -P ./cartoonGAN/model_ckpt

cd ..
