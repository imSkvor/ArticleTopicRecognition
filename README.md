# ArticleTopicRecognition


### Конвертация -> ONNX -> Tensor-RT

В `ONNX`:
```bash
optimum-cli export onnx \
    --model ./output/checkpoints \
    --task text-classification \
    ./output/onnx
```

В `Tensor-TR` (tensorrt-утилита весит... 10Gb :) ):
```bash
polygraphy convert ./output/onnx/model.onnx \
    --engine ./models/model.engine \
    --builder-opt-level 3 \
    --fp16 \
    --min-shapes input_ids:1x1,attention_mask:1x1 \
    --opt-shapes input_ids:1x64,attention_mask:1x64 \
    --max-shapes input_ids:8x256,attention_mask:8x256
```
