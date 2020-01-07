conda activate decent

freeze_graph --input_graph=./project/model.pb \
             --input_checkpoint=./project/model.ckpt \
             --input_binary=true \
             --output_graph=./project/frozen_model.pb \
             --output_node_names=y_pred