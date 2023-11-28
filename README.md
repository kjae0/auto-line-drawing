# line-drawing style transfer

Style transfer from original image to line-drawing art.
There are 3 steps for generation, 1) cartoonize, 2) segmentation mask generation, 3) generate line-drawing art.

### End-to-End Generation from origin image
```
  sh run.sh
```

### Cartoonize
```
  sh run_cartoonGAN.sh
```

### Segment Anything
```
  sh run_SAM.sh
```

### Line Drawing
```
  sh run_SAM.sh
```

## Reference
CartoonGAN: https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch <br>
Segment Anything: https://github.com/facebookresearch/segment-anything
