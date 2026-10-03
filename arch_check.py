import os, re

root = r'src\gateway'
patterns = ['provider == ', 'provider_name ==', '"openai"', '"anthropic"', '"gemini"']
violations = []
for dirpath, dirnames, filenames in os.walk(root):
    if 'adapters' in dirpath:
        continue
    for f in filenames:
        if not f.endswith('.py'):
            continue
        path = os.path.join(dirpath, f)
        lines = open(path, encoding='utf-8').readlines()
        for i, line in enumerate(lines, 1):
            for pat in patterns:
                if pat in line:
                    violations.append(f'{path}:{i}: {line.rstrip()}')

if violations:
    print('ARCHITECTURE VIOLATIONS:')
    for v in violations:
        print(v)
else:
    print('CLEAN: No architecture violations found outside adapters/')
