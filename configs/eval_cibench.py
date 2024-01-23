from lagent.agents.react import ReActProtocol
from mmengine.config import read_base

from opencompass.models.lagent import CodeAgent
from opencompass.models import HuggingFaceCausalLM
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask

from cibench.lagent.actions.ipython_interpreter import IPythonInterpreter
from cibench.lagent.agents.react import CIReAct

with read_base():
    from .datasets.CIBench.CIBench_template_gen_e6b12a import cibench_datasets as cibench_datasets_template
    from .datasets.CIBench.CIBench_generation_gen_8ab0dc import cibench_datasets as cibench_datasets_generation
    from .datasets.CIBench.CIBench_template_wgt_gen_fecda1 import cibench_datasets as cibench_datasets_template_wgt
    from .datasets.CIBench.CIBench_generation_wgt_gen_c4a7c1 import cibench_datasets as cibench_datasets_generation_wgt
    from .summarizers.cibench import summarizer

datasets = []
datasets += cibench_datasets_template
datasets += cibench_datasets_generation
datasets += cibench_datasets_template_wgt
datasets += cibench_datasets_generation_wgt

FORCE_STOP_PROMPT_EN = """You should directly give results based on history information."""

FEWSHOT_INSTRUCTION = """\
You are an assistant who can utilize external tools.
{tool_description}
To use a tool, please response with the following format:
```
{thought} Think what you need to solve, do you need to use tools?
{action} The tool name, should be one of [{action_names}].
{action_input} The input to the tool that you want to use.
```
The tool will give you response after your response using the following format:
```
{response} the results after call the tool.
```
Therefore DO NOT generate tool response by yourself.

Also please follow the guidelines:
1. Always use code interpreter to solve the problem.
2. The generated codes should always in a markdown code block format.
3. The generated codes will be executed in an ipython manner and the results will be cached.
4. Your responded code should always be simple and only solves the problem in current step.

For example:

File url: `xxxx`
### Step 1. Load the dataset from the url into a pandas DataFrame named `df`.

{thought} We should use `pandas` to solve this step.
{action} IPythonInterpreter
{action_input} ```python
import pandas as pd
url = "xxxx"
data = pd.read_csv(url)
```
{response} The code is succeed without any outputs.

Let us begin from here!
"""

IPYTHON_INTERPRETER_DESCRIPTION = '''\
It can run Python code in a manner as jupyter notebook. The code must be a valid code that contains only python method.'''

_meta_template = dict(
    round=[
        dict(role='HUMAN', begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role='SYSTEM', begin='<|im_start|>system\n', end='<|im_end|>\n'),
        dict(role='BOT', begin='<|im_start|>assistant\n', end='<|im_end|>\n', generate=True),
    ],
    eos_token_id=92542
)


actions=[dict(type=IPythonInterpreter, user_data_dir='./data/cibench_dataset/datasources',
                 description=IPYTHON_INTERPRETER_DESCRIPTION)]
protocol=dict(
            type=ReActProtocol,
            call_protocol=FEWSHOT_INSTRUCTION,
            force_stop=FORCE_STOP_PROMPT_EN,
            finish=dict(role='FINISH', begin='Final Answer:', end='\n'),
        )
models = [
    dict(
        abbr="internlm2-chat-7b-hf",
        type=CodeAgent,
        agent_type=CIReAct,
        llm=dict(
            type=HuggingFaceCausalLM,
            path="internlm/internlm2-chat-7b",
            tokenizer_path="internlm/internlm2-chat-7b",
            tokenizer_kwargs=dict(
                padding_side="left",
                truncation_side="left",
                use_fast=False,
                trust_remote_code=True,
            ),
            max_seq_len=2048,
            model_kwargs=dict(trust_remote_code=True, device_map="auto"),
        ),
        protocol=protocol,
        actions=actions,
        max_turn=3,
        run_cfg=dict(num_gpus=1, num_procs=1),
        batch_size=8,
        meta_template=_meta_template,
        end_str='<|im_end|>',
    ),
]


infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16,
        task=dict(type=OpenICLInferTask)),
)
