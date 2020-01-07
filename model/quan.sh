#!/bin/bash


# activate DECENT_Q Python3.6 virtual environment
conda activate decent_q3

# generate calibraion images and list file
#python generate_images.py

# remove existing files
rm -rf ./quantize_results


# run quantization
echo "#####################################"
echo "QUANTIZE"
echo "#####################################"
decent_q quantize \
  --input_frozen_graph ./project/frozen_model.pb \
  --input_nodes x \
  --input_shapes ?,64,64,3 \
  --output_nodes y_pred \
  --method 1 \
  --input_fn input_fn.calib_input \
  --calib_iter 799

echo "#####################################"
echo "QUANTIZATION COMPLETED"
echo "#####################################"