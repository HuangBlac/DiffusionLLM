import ast
import base64
import hashlib
import io
import zipfile

from experiments import build_learning_run_notebook as builder


def test_locked_notebook_does_not_import_cloud_or_train(capsys):
    notebook = builder.build()
    scope = {}
    for cell in notebook['cells']:
        exec(''.join(cell['source']), scope)
    assert scope['RUN'] is False and scope['RUN_CPU_AUTH'] is False
    assert 'runtime' not in scope
    assert 'Learning run locked' in capsys.readouterr().out
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(scope['BUNDLE']))) as archive:
        assert set(archive.namelist()) == set(builder.SOURCES)
        for name, digest in scope['SOURCE_HASHES'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest


def test_proof_excludes_reserved_splits_and_config_stays_small():
    proof = {key: {split: (4096 if split == 'training' else 100 if split == 'development' else 300)
                   for split in ('training', 'development', 'reserved', 'calibration')}
             for key in ('counts', 'artifacts', 'artifact_files')}
    proof['tokenizer_sha256'] = 'verified-tokenizer'
    notebook = builder.build(proof, 'test-parent')
    scope = {}
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        ast.parse(source)
        exec(source, scope)
    assert all(set(scope['INPUT_PROOF'][key]) == {'training', 'development'}
               for key in ('counts', 'artifacts', 'artifact_files'))
    config = ''.join(notebook['cells'][1]['source'])
    assert len(config) < 1000 and 'BUNDLE' not in config
