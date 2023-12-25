import os
import torch
from typing import Union, Dict, Any
from modelscope.pipelines.builder import PIPELINES
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.nlp.text_generation_pipeline import TextGenerationPipeline
from modelscope.models.base import Model, TorchModel
from modelscope.utils.logger import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer

@PIPELINES.register_module(Tasks.text_generation, module_name='Baichuan-7B-text-generation-pipe')
class Baichuan7BTextGenerationPipeline(TextGenerationPipeline):
    def __init__(
            self,
            model: Union[Model, str],
            *args,
            **kwargs):
        model = Baichuan7BTextGeneration(model, **kwargs) if isinstance(model, str) else model
        super().__init__(model=model, **kwargs)
    
    def preprocess(self, inputs, **preprocess_params) -> Dict[str, Any]:
        return inputs
    
    def _sanitize_parameters(self, **pipeline_parameters):
        return {},pipeline_parameters,{}
    
    # define the forward pass
    def forward(self, inputs: Dict, **forward_params) -> Dict[str, Any]:
        return self.model(inputs,**forward_params)
    
    # format the outputs from pipeline
    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        return input


@MODELS.register_module(Tasks.text_generation, module_name='Baichuan-7B')
class Baichuan7BTextGeneration(TorchModel):
    def __init__(self, model_dir=None, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.logger = get_logger()
        # loading tokenizer
        device_map = kwargs.get('device_map', 'auto')
        torch_dtype = kwargs.get('torch_dtype', torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, device_map=device_map, 
                                                          torch_dtype=torch_dtype, 
                                                          trust_remote_code=True)
        self.model = self.model.eval()

    def forward(self,input: Dict, *args, **kwargs) -> Dict[str, Any]:
        output = {}
        res = self.infer(input,**kwargs)
        output['text'] = res
        return output
    
    def quantize(self, bits: int):
        self.model = self.model.quantize(bits)
        return self
    
    def infer(self, input, **kwargs):
        device = self.model.device
        input_ids = self.tokenizer(input, return_tensors="pt").input_ids.to(device)
        pred = self.model.generate(input_ids,**kwargs)
        out = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        return out