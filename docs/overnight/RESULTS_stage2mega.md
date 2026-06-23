# Stage2 메가서치 (600조합 프록시 4-shard 병렬 → 상위 12 풀학습)

> best 단일모델 test macro-F1 **0.9274** (val 0.9232)  ·  6-앙상블+보정 **0.935**가 전체 best 유지

## Phase2 풀학습(test macro-F1)

| test F1 | val F1 | arch | width | pool | dropout | lr | wd | loss | mixup |
|---|---|---|---|---|---|---|---|---|---|
| 0.9274 | 0.9232 | resnet | 48 | maxavg | 0.1 | 0.001449 | 1e-05 | bce | none |
| 0.9272 | 0.9234 | resnet_cbam | 48 | maxavg | 0.1 | 0.001631 | 0.001 | bce | none |
| 0.9242 | 0.9220 | resnet | 64 | maxavg | 0.2 | 0.001231 | 0.0001 | bce | none |
| 0.9220 | 0.9148 | resnet | 48 | maxavg | 0.2 | 0.000809 | 1e-05 | bce | none |
| 0.9203 | 0.9111 | resnet_cbam | 64 | gem | 0.2 | 0.000744 | 0.001 | bce | none |
| 0.9200 | 0.9242 | resnet | 64 | gem | 0.1 | 0.000766 | 0.001 | focal | none |
| 0.9146 | 0.9190 | resnet | 64 | maxavg | 0.2 | 0.000376 | 0.0001 | bce | none |
| 0.9106 | 0.9220 | resnet | 64 | maxavg | 0.4 | 0.000685 | 0.0001 | bce | none |
| 0.9091 | 0.9192 | resnet | 64 | maxavg | 0.4 | 0.000423 | 0.0003 | bce | none |
| 0.9040 | 0.9166 | resnet | 64 | maxavg | 0.3 | 0.000909 | 0.0001 | bce | none |
| 0.9027 | 0.8976 | resnet_cbam | 48 | maxavg | 0.1 | 0.001631 | 0.001 | bce | cutmix |
| 0.9003 | 0.8988 | resnet | 48 | maxavg | 0.1 | 0.001449 | 1e-05 | bce | mixup |
| 0.9003 | 0.9000 | resnet_cbam | 64 | gem | 0.3 | 0.000458 | 0.001 | asl | none |
| 0.8935 | 0.9003 | resnet | 48 | maxavg | 0.1 | 0.001449 | 1e-05 | bce | cutmix |
| 0.8899 | 0.9002 | resnet_cbam | 48 | maxavg | 0.1 | 0.001631 | 0.001 | bce | mixup |
| 0.8847 | 0.8951 | resnet | 32 | maxavg | 0.1 | 0.001296 | 1e-05 | asl | none |

## Phase1 프록시 top25

| proxy val F1 | cfg |
|---|---|
| 0.8869 | {'arch': 'resnet', 'width': 48, 'pool': 'maxavg', 'dropout': 0.1, 'lr': 0.001449, 'wd': 1e-05, 'loss': 'bce'} |
| 0.8753 | {'arch': 'resnet', 'width': 48, 'pool': 'maxavg', 'dropout': 0.2, 'lr': 0.000809, 'wd': 1e-05, 'loss': 'bce'} |
| 0.8742 | {'arch': 'resnet', 'width': 64, 'pool': 'maxavg', 'dropout': 0.4, 'lr': 0.000423, 'wd': 0.0003, 'loss': 'bce'} |
| 0.8732 | {'arch': 'resnet_cbam', 'width': 64, 'pool': 'gem', 'dropout': 0.3, 'lr': 0.000458, 'wd': 0.001, 'loss': 'asl'} |
| 0.8727 | {'arch': 'resnet', 'width': 64, 'pool': 'gem', 'dropout': 0.1, 'lr': 0.000766, 'wd': 0.001, 'loss': 'focal'} |
| 0.8723 | {'arch': 'resnet_cbam', 'width': 48, 'pool': 'maxavg', 'dropout': 0.1, 'lr': 0.001631, 'wd': 0.001, 'loss': 'bce'} |
| 0.8720 | {'arch': 'resnet', 'width': 64, 'pool': 'maxavg', 'dropout': 0.3, 'lr': 0.000909, 'wd': 0.0001, 'loss': 'bce'} |
| 0.8716 | {'arch': 'resnet', 'width': 64, 'pool': 'maxavg', 'dropout': 0.4, 'lr': 0.000685, 'wd': 0.0001, 'loss': 'bce'} |
| 0.8696 | {'arch': 'resnet', 'width': 64, 'pool': 'maxavg', 'dropout': 0.2, 'lr': 0.000376, 'wd': 0.0001, 'loss': 'bce'} |
| 0.8685 | {'arch': 'resnet', 'width': 32, 'pool': 'maxavg', 'dropout': 0.1, 'lr': 0.001296, 'wd': 1e-05, 'loss': 'asl'} |
| 0.8658 | {'arch': 'resnet_cbam', 'width': 64, 'pool': 'gem', 'dropout': 0.2, 'lr': 0.000744, 'wd': 0.001, 'loss': 'bce'} |
| 0.8653 | {'arch': 'resnet', 'width': 64, 'pool': 'maxavg', 'dropout': 0.2, 'lr': 0.001231, 'wd': 0.0001, 'loss': 'bce'} |
| 0.8650 | {'arch': 'resnet', 'width': 32, 'pool': 'gem', 'dropout': 0.2, 'lr': 0.00191, 'wd': 0.001, 'loss': 'asl'} |
| 0.8617 | {'arch': 'resnet', 'width': 64, 'pool': 'gem', 'dropout': 0.3, 'lr': 0.000488, 'wd': 0.003, 'loss': 'focal'} |
| 0.8617 | {'arch': 'resnet', 'width': 32, 'pool': 'gem', 'dropout': 0.2, 'lr': 0.000665, 'wd': 0.0001, 'loss': 'asl'} |
| 0.8614 | {'arch': 'resnet', 'width': 64, 'pool': 'maxavg', 'dropout': 0.1, 'lr': 0.002363, 'wd': 0.0001, 'loss': 'bce'} |
| 0.8610 | {'arch': 'resnet_cbam', 'width': 64, 'pool': 'gem', 'dropout': 0.3, 'lr': 0.000434, 'wd': 1e-05, 'loss': 'bce'} |
| 0.8607 | {'arch': 'cnn', 'width': 48, 'pool': 'gem', 'dropout': 0.1, 'lr': 0.001122, 'wd': 1e-05, 'loss': 'focal'} |
| 0.8601 | {'arch': 'resnet_cbam', 'width': 48, 'pool': 'maxavg', 'dropout': 0.1, 'lr': 0.000622, 'wd': 0.0003, 'loss': 'focal'} |
| 0.8594 | {'arch': 'cnn', 'width': 64, 'pool': 'maxavg', 'dropout': 0.1, 'lr': 0.000584, 'wd': 0.001, 'loss': 'bce'} |
| 0.8594 | {'arch': 'resnet', 'width': 32, 'pool': 'maxavg', 'dropout': 0.1, 'lr': 0.001, 'wd': 0.0001, 'loss': 'asl'} |
| 0.8593 | {'arch': 'resnet', 'width': 48, 'pool': 'maxavg', 'dropout': 0.2, 'lr': 0.000465, 'wd': 0.003, 'loss': 'bce'} |
| 0.8593 | {'arch': 'resnet_cbam', 'width': 48, 'pool': 'maxavg', 'dropout': 0.3, 'lr': 0.000609, 'wd': 0.001, 'loss': 'asl'} |
| 0.8575 | {'arch': 'resnet', 'width': 64, 'pool': 'gem', 'dropout': 0.2, 'lr': 0.000364, 'wd': 0.003, 'loss': 'focal'} |
| 0.8574 | {'arch': 'resnet', 'width': 32, 'pool': 'gem', 'dropout': 0.1, 'lr': 0.00076, 'wd': 0.001, 'loss': 'asl'} |
