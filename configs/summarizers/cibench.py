from mmengine.config import read_base

with read_base():
    from .groups.cibench import cibench_summary_groups

summarizer = dict(
    dataset_abbrs=[
        # '######## CIBench Generation########', # category
        # 'cibench_generation:tool_rate',
        # 'cibench_generation:executable',
        # 'cibench_generation:numeric_correct',
        # 'cibench_generation:text_score',
        # 'cibench_generation:vis_sim',
        # '######## CIBench Generation With GT########', # category
        # 'cibench_generation_wgt:tool_rate',
        # 'cibench_generation_wgt:executable',
        # 'cibench_generation_wgt:numeric_correct',
        # 'cibench_generation_wgt:text_score',
        # 'cibench_generation_wgt:vis_sim',
        # '######## CIBench Template ########', # category
        # 'cibench_template:tool_rate',
        # 'cibench_template:executable',
        # 'cibench_template:numeric_correct',
        # 'cibench_template:text_score',
        # 'cibench_template:vis_sim',
        # '######## CIBench Template With GT########', # category
        # 'cibench_template_wgt:tool_rate',
        # 'cibench_template_wgt:executable',
        # 'cibench_template_wgt:numeric_correct',
        # 'cibench_template_wgt:text_score',
        # 'cibench_template_wgt:vis_sim',
        # '######## CIBench Template Chinese ########', # category
        # 'cibench_template_cn:tool_rate',
        # 'cibench_template_cn:executable',
        # 'cibench_template_cn:numeric_correct',
        # 'cibench_template_cn:text_score',
        # 'cibench_template_cn:vis_sim',
        # '######## CIBench Template Chinese With GT########', # category
        # 'cibench_template_cn_wgt:tool_rate',
        # 'cibench_template_cn_wgt:executable',
        # 'cibench_template_cn_wgt:numeric_correct',
        # 'cibench_template_cn_wgt:text_score',
        # 'cibench_template_cn_wgt:vis_sim',
        '######## CIBench Abstrct Metric ########', 
        'cibench_data_manipulation:scores',
        'cibench_data_visualization:scores',
        'cibench_modeling:scores',
        'cibench_nlp:scores',
        'cibench_ip:scores',
        'cibench_math:scores',
        '######## CIBench Abstrct Metric With GT ########', 
        'cibench_data_manipulation_wgt:scores',
        'cibench_data_visualization_wgt:scores',
        'cibench_modeling_wgt:scores',
        'cibench_nlp_wgt:scores',
        'cibench_ip_wgt:scores',
        'cibench_math_wgt:scores',

    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith("_summary_groups")], [])
)
