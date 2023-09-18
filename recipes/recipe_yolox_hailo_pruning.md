<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

---

version: 1.1.0

# General Hyperparams
num_epochs: 400
init_lr: 0.00005
final_lr: 0.000005
weights_warmup_lr: 0
biases_warmup_lr: 0

# Pruning Hyperparams
init_sparsity: 0.05
pruning_start_epoch: 295
pruning_end_epoch: 345
pruning_update_frequency: 2.0

#Modifiers
training_modifiers:
  - !EpochRangeModifier
    start_epoch: 0
    end_epoch: eval(num_epochs)
    
  - !LearningRateFunctionModifier
    start_epoch: 3
    end_epoch: eval(num_epochs)
    lr_func: linear
    init_lr: eval(init_lr)
    final_lr: eval(final_lr)
    
  - !LearningRateFunctionModifier
    start_epoch: 0
    end_epoch: 3
    lr_func: linear
    init_lr: eval(weights_warmup_lr)
    final_lr: eval(init_lr)
    param_groups: [0, 1]
    
  - !LearningRateFunctionModifier
    start_epoch: 0
    end_epoch: 3
    lr_func: linear
    init_lr: eval(biases_warmup_lr)
    final_lr: eval(init_lr)
    param_groups: [2]

pruning_modifiers:
  - !GMPruningModifier
    params:
      # - backbone.backbone.stage0.rbr_dense.conv.weight
      - re:backbone.backbone.*.*.rbr_dense.conv.weight
      - re:backbone.neck.*.*.rbr_dense.conv.weight
      # - re:backbone.backbone.*.weight
      # - re:backbone.neck.*.weight
    init_sparsity: eval(init_sparsity)
    final_sparsity: eval(0.7)
    start_epoch: eval(pruning_start_epoch)
    end_epoch: eval(pruning_end_epoch)
    update_frequency: eval(pruning_update_frequency)
---

# YOLOv5s pruned -- COCO

This recipe lists the hyper-parameters used to prune the YOLOv5s model using the COCO dataset. The model is pruned to an overall sparsity of 65%. To vary hyperparams either edit the recipe or supply the --recipe_args argument to the training commands.
For example, the following appended to the training commands will change the number of epochs:
```bash
--recipe_args '{"num_epochs":150}'
```

## Training

To set up the training environment, [install SparseML with PyTorch](https://github.com/neuralmagic/sparseml#installation)

The following command is used to prune the YOLOv5s model on the COCO dataset using 2 GPUs.

```bash
python -m torch.distributed.run --no_python --nproc_per_node 2 \
sparseml.yolov5.train \
  --cfg yolov5s.yaml \
  --data coco.yaml \
  --batch-size 64 \
  --recipe zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65-none \
  --hyp hyps/hyp.scratch-low.yaml \
  --weights zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none \
  --patience 0 \
  --gradient-accum-steps 4
```

## Evaluation

This model achieves 56.3 mAP@0.5 on the COCO dataset. The following command can be used to validate accuracy.

```bash
sparseml.yolov5.validation \
  --weights "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned65-none" \
  --data coco.yaml \
  --iou-thres 0.65
```
