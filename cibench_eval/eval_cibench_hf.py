from copy import deepcopy
from mmengine.config import read_base
from opencompass.models.lagent import LagentAgent
from lagent import ReAct
from lagent.agents.react import ReActProtocol
from opencompass.models.lagent import CodeAgent
from opencompass.lagent.actions.python_interpreter import PythonInterpreter
from opencompass.lagent.actions.ipython_interpreter import IPythonInterpreter
from opencompass.lagent.agents.react import CIReAct
from opencompass.models import HuggingFaceCausalLM
from opencompass.partitioners import SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask

with read_base():
    # You can import your models or any other model here
    from ..models.hf_internlm.hf_internlm2_chat_20b import models as internlm2_chat_20b_model
    from .prompts import FEWSHOT_INSTRUCTION, FORCE_STOP_PROMPT_EN, IPYTHON_INTERPRETER_DESCRIPTION

    from ..summarizers.cibench import summarizer

    from ..datasets.CIBench.CIBench_template_gen_e6b12a import cibench_datasets as cibench_datasets_template
    from ..datasets.CIBench.CIBench_generation_gen_8ab0dc import cibench_datasets as cibench_datasets_generation
    from ..datasets.CIBench.CIBench_template_oracle_gen_fecda1 import cibench_datasets as cibench_datasets_template_oracle
    from ..datasets.CIBench.CIBench_generation_oracle_gen_c4a7c1 import cibench_datasets as cibench_datasets_generation_oracle

_origin_models = sum([v for k, v in locals().items() if k.endswith("_model")], [])

datasets = []
datasets += cibench_datasets_template
datasets += cibench_datasets_generation
datasets += cibench_datasets_template_oracle
datasets += cibench_datasets_generation_oracle
work_dir = './outputs/cibench/'

_agent_models = []
for m in _origin_models:
    m = deepcopy(m)
    if 'meta_template' in m and 'round' in m['meta_template']:
        round = m['meta_template']['round']
        if all(r['role'].upper() != 'SYSTEM' for r in round):  # no system round
            if not any('api_role' in r for r in round):
                m['meta_template']['round'].append(dict(role="system", begin="System response:", end="\n"))
            else:
                m['meta_template']['round'].append(dict(role="system", api_role="SYSTEM"))
            print(f'WARNING: adding SYSTEM round in meta_template for {m.get("abbr", None)}')
    _agent_models.append(m)

protocol=dict(
    type=ReActProtocol,
    call_protocol=FEWSHOT_INSTRUCTION,
    force_stop=FORCE_STOP_PROMPT_EN,
    finish=dict(role='FINISH', begin='Final Answer:', end='\n'),
)

models = []
for m in _agent_models:
    m = deepcopy(m)
    origin_abbr = m.pop('abbr')
    abbr = origin_abbr + '-cibench-react'
    m.pop('batch_size', None)
    m.pop('max_out_len', None)
    m.pop('max_seq_len', None)
    run_cfg = m.pop('run_cfg', {})

    agent_model = dict(
        abbr=abbr,
        summarizer_abbr=origin_abbr,
        type=CodeAgent,
        agent_type=CIReAct,
        max_turn=3,
        llm=m,
        actions=[dict(type=IPythonInterpreter, user_data_dir='./data/cibench_dataset/datasources', description=IPYTHON_INTERPRETER_DESCRIPTION)],
        protocol=protocol,
        batch_size=1,
        run_cfg=run_cfg,
    )
    models.append(agent_model)

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, strategy='split'),
    runner=dict(
        type=SlurmSequentialRunner,
        max_num_workers=64,
        quotatype='reserved',
        partition='llmeval',
        task=dict(type=OpenICLInferTask)),
)

# an example for local runner
# infer = dict(
#     partitioner=dict(type=SizePartitioner, max_task_size=1000, strategy='split'),
#     runner=dict(
#         type=LocalRunner,
#         max_num_workers=64,
#         task=dict(type=OpenICLInferTask)),
# )