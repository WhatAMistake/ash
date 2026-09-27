import ast
from pathlib import Path
from types import SimpleNamespace


def test_keyword_fallback_filters_editorial():
    # Load the real fallback without importing optional embedding libraries.
    source = Path(__file__).resolve().parents[1] / 'src' / 'writer_rag.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WriterRAG')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_keyword_fallback')
    ns = {'Optional': __import__('typing').Optional, 'RAGResult': SimpleNamespace}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), ns)
    bot = SimpleNamespace(all_chunks=[{'text': 'writing craft'}], craft_chunks=[], style_chunks=[],
                          editorial_chunks=[{'text': 'writing editorial'}])
    result = ns['_keyword_fallback'](bot, 'writing', 3, 'editorial')
    assert [r.content for r in result] == ['writing editorial']
