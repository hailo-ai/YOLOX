
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
      - re:backbone.backbone.*.*.rbr_dense.conv.weight
      - re:backbone.neck.*.*.rbr_dense.conv.weight
    init_sparsity: eval(init_sparsity)
    final_sparsity: eval(0.7)
    start_epoch: eval(pruning_start_epoch)
    end_epoch: eval(pruning_end_epoch)
    update_frequency: eval(pruning_update_frequency)
---
