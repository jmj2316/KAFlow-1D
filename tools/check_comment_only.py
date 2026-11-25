"""Check that edits to Python files changed only comments and string literals (e.g. translated docstrings and messages).
For each file, the AST with every string constant blanked must equal that of the committed version (git HEAD); also reports remaining Hangul.
  python tools/check_comment_only.py FILE [FILE ...]
"""
import ast, re, subprocess, sys


class Blank(ast.NodeTransformer):
    def visit_Constant(self, node):
        return ast.copy_location(ast.Constant(value=''), node) if isinstance(node.value, str) else node


def norm(src):
    return ast.dump(Blank().visit(ast.parse(src)), include_attributes=False)


ok = True
for f in sys.argv[1:]:
    new = open(f, encoding='utf-8').read()
    old = subprocess.run(['git', 'show', f'HEAD:{f}'], capture_output=True, text=True).stdout
    same = norm(old) == norm(new)
    hangul = [i + 1 for i, line in enumerate(new.splitlines()) if re.search('[\\uac00-\\ud7a3]', line)]
    compile(new, f, 'exec')
    ok &= same and not hangul
    print(f"{'OK ' if same else 'CODE CHANGED'} {f}" + (f'  Hangul on lines {hangul}' if hangul else ''))
sys.exit(0 if ok else 1)
