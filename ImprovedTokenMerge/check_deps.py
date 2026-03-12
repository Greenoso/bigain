import re
import os

files = [f for f in os.listdir('.') if f.endswith('.py') and not f.startswith('test_') and not f.startswith('check_')]

deps = {}
for f in files:
    with open(f, 'r') as fh:
        content = fh.read()
        # Find local imports
        imports = re.findall(r'from \.([\w_]+) import|from ([\w_]+) import', content)
        local_imports = set()
        for match in imports:
            mod = match[0] or match[1]
            if mod and f'{mod}.py' in files:
                local_imports.add(f'{mod}.py')
        deps[f] = sorted(local_imports)

print("File dependencies:")
for f in sorted(deps.keys()):
    if deps[f]:
        print(f"{f}: {', '.join(deps[f])}")
    else:
        print(f"{f}: (no local deps)")
