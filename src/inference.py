import json
from typing import Optional, Union

import tensorrt as trt
import torch
from torch import Tensor
from transformers import AutoTokenizer

from arxiv_taxonomy import arxiv_category_names

class TRTClassifier:

    def __init__(
        self,
        engine_path: str,
        label_mapping_path: str,
        checkpoint_dir: str,
    ) -> None:

        self._logger: trt.Logger = trt.Logger(trt.Logger.ERROR)

        with open(engine_path, "rb") as f:
            with trt.Runtime(self._logger) as runtime:
                self._engine: trt.ICudaEngine = runtime.deserialize_cuda_engine(f.read())

        self._context: trt.IExecutionContext = self._engine.create_execution_context()

        self._input_names: list[str] = []
        self._output_name: str = ""

        for index in range(self._engine.num_io_tensors):
            tensor_name: str = self._engine.get_tensor_name(index)
            tensor_mode: trt.TensorIOMode = self._engine.get_tensor_mode(tensor_name)

            if tensor_mode == trt.TensorIOMode.INPUT:
                self._input_names.append(tensor_name)
            elif tensor_mode == trt.TensorIOMode.OUTPUT:
                self._output_name = tensor_name

        # load labels
        with open(label_mapping_path, "r", encoding = "utf-8") as f:
            mapping_data: dict[str, dict[str, int]] = json.load(f)

        self._id2label: dict[int, str] = {
            int(idx): label for label, idx in mapping_data["label2id"].items()
        }

        self._tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self._stream: torch.cuda.Stream = torch.cuda.Stream()

    def predict(
        self,
        title: str,
        abstract: str,
        top_k: Optional[int] = None,
        cumulative_threshold: float = 0.95,
    ) -> list[dict[str, Union[float, str]]]:

        text: str = f"{title.strip()} {abstract.strip()}" if abstract else title.strip()

        encoded: dict[str, Tensor] = self._tokenizer(
            text,
            return_tensors = "pt",
            truncation = True,
            max_length = 256,
            padding = "max_length",
        )

        input_ids: Tensor = encoded["input_ids"].cuda().to(dtype = torch.int64)
        attention_mask: Tensor = encoded["attention_mask"].cuda().to(dtype = torch.int64)

        batch_size: int = input_ids.size(0)
        seq_length: int = input_ids.size(1)

        for input_name in self._input_names:
            self._context.set_input_shape(input_name, (batch_size, seq_length))

        if "token_type_ids" in self._input_names:
            token_type_ids: Tensor = torch.zeros(
                (batch_size, seq_length), dtype = torch.int64, device = "cuda"
            )
            self._context.set_tensor_address("token_type_ids", token_type_ids.data_ptr())

        self._context.set_tensor_address("input_ids", input_ids.data_ptr())
        self._context.set_tensor_address("attention_mask", attention_mask.data_ptr())

        # for input_name in self._input_names:
        #     if input_name == "input_ids":
        #         self._context.set_input_shape(input_name, (batch_size, seq_length))
        #     elif input_name == "attention_mask":
        #         self._context.set_input_shape(input_name, (batch_size, seq_length))
        #     elif input_name == "token_type_ids":
        #         token_type_ids: Tensor = torch.zeros(
        #             (batch_size, seq_length), dtype = torch.int32, device = "cuda"
        #         )
        #         self._context.set_tensor_address(input_name, token_type_ids.data_ptr())

        output_shape: list[int] = self._context.get_tensor_shape(self._output_name)
        if output_shape[0] == -1:
            output_shape[0] = batch_size

        output_tensor: Tensor = torch.empty(
            tuple(output_shape), dtype = torch.float32, device = "cuda"
        )

        self._context.set_tensor_address("input_ids", input_ids.data_ptr())
        self._context.set_tensor_address("attention_mask", attention_mask.data_ptr())
        self._context.set_tensor_address(self._output_name, output_tensor.data_ptr())

        # inference itself
        with torch.cuda.stream(self._stream):
            self._context.execute_async_v3(self._stream.cuda_stream)
            self._stream.synchronize()

        logits: Tensor = output_tensor
        # print(f"Raw logits (first 5): {logits[0, :5].cpu().tolist()}")
        probabilities: Tensor = torch.softmax(logits, dim = -1).squeeze(0).cpu()

        return self._extract_top_classes(
            probabilities = probabilities,
            top_k = top_k,
            cumulative_threshold = cumulative_threshold,
        )

    def _extract_top_classes(
        self,
        probabilities: Tensor,
        top_k: Optional[int],
        cumulative_threshold: float,
    ) -> list[dict[str, Union[str, float]]]: #-> list[tuple[str, float]]:

        sorted_indices: Tensor = torch.argsort(probabilities, descending = True)

        result: list[dict[str, Union[str, float]]] = []
        cumulative_sum: float = 0

        for rank, idx in enumerate(sorted_indices.tolist()):
            if top_k is not None and rank >= top_k:
                break

            prob: float = probabilities[idx].item()
            cumulative_sum += prob

            label_code: str = self._id2label.get(idx, f"unknown_{idx}")
            label_info = self._format_label(label_code)

            result.append({
                "code": label_info["code"],
                "name": label_info["name"],
                "display": label_info["display"],
                "probability": prob,
            })

            if cumulative_sum >= cumulative_threshold:
                break

        return result
    
    def _format_label(self, code: str) -> dict[str, str]:
        name = arxiv_category_names.get(code, code)
        return {
            "code": code,
            "name": name,
            "display": f"{name} ({code})"
        }
