"""Compare Python syntax across interpreter versions; omit docstrings and locations."""
import ast,hashlib,json
from pathlib import Path

def canonical(node):
    if isinstance(node,ast.AST):
        fields={k:v for k,v in ast.iter_fields(node) if k!='type_params'}
        if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            b=fields.get('body',[])
            if b and isinstance(b[0],ast.Expr) and isinstance(b[0].value,ast.Constant) and isinstance(b[0].value.value,str):fields['body']=b[1:]
        return {'node':type(node).__name__,**{k:canonical(v) for k,v in fields.items() if v is not None and v!=[]}}
    if isinstance(node,list):return [canonical(x) for x in node]
    if isinstance(node,bytes):return {'bytes_hex':node.hex()}
    return node

def code_hash(path):
    raw=json.dumps(canonical(ast.parse(Path(path).read_text(encoding='utf-8'))),sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(raw).hexdigest()

if __name__=='__main__':
    print(json.dumps({n:code_hash(n) for n in ['geometry.py','convert_geometry.py','experiment.py','screen.py','stability.py','guarded.py']}))
