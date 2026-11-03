from experiments.colab_pipeline import p1_stages
from experiments.build_colab_notebook import build
import ast
import math


def test_p1_keeps_same_output_and_entry_for_resume(tmp_path):
    stages = {s.name: s for s in p1_stages(tmp_path)}
    for method in ('ar', 'dlm'):
        first, last = stages[f'{method}-20'], stages[f'{method}-200']
        assert '--resume' not in first.command and '--resume' in last.command
        assert first.checkpoint == last.checkpoint
        assert first.command[first.command.index('--output')+1] == last.command[last.command.index('--output')+1]
        assert last.expected == {'start_step': 20, 'end_step': 200}
        assert 'experiments.full_parameter_pilot' in first.command
        assert 'experiments.full_parameter_pilot' in last.command
    assert '--evaluate-only' in stages['original-ar'].command
    assert stages['original-ar'].checkpoint is None


def test_p1_wall_limit_reserves_half_a_compute_unit():
    source = ''.join(build()['cells'][1]['source'])
    tree = ast.parse(source)
    constants = {n.targets[0].id: n.value.value for n in tree.body
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and isinstance(n.value, ast.Constant)}
    assert constants['RELEASE_RESERVE_CU'] == .5
    value = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'time_cap')
    seconds = eval(compile(ast.Expression(value), '<budget-test>', 'eval'),
                   dict(MODE='p1', CAP_CU=4, SPENT_CU=0, RATE_CU_HOUR=6.73,
                        RELEASE_RESERVE_CU=constants['RELEASE_RESERVE_CU'], floor=math.floor))
    assert seconds * 6.73 / 3600 <= 3.5
