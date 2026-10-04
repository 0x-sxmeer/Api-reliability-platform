import os
import re
import sys

# Portable path: works on both Windows and POSIX checkouts.
root = os.path.join("src", "gateway")

# The One Rule: vendor-specific KNOWLEDGE (branching logic) must live only in
# adapters/. We flag real code constructs, not mere string occurrences:
#   - `provider == ...` / `provider_name == ...` comparisons
#   - quoted vendor names used as values/keys in *code* lines
# Comments/docstrings are stripped first; declarative data tables (pricing.py)
# are an explicitly reviewed allow-list because they are keyed by each adapter's
# own provider_name and contain zero branching logic.
VENDOR_NAME = re.compile(r"""["'](openai|anthropic|gemini)["']""", re.IGNORECASE)
COMPARISON = re.compile(r"(?<!\w)provider(?:_name)?\s*==")
ALLOWLIST = {os.path.join("src", "gateway", "core", "pricing.py")}

# Reviewed exemptions — each entry is a deliberate, documented exception to the
# regex gate, NOT an oversight. Keep this list honest: adding a file here is an
# architecture decision that belongs in docs/interface-changes.md.
#
# devmode.py: fake transport that routes canned responses by URL host. It holds
# no reliability/governance knowledge — it is test scaffolding at the transport
# layer, and its host strings mirror the base URLs already declared inside
# adapters/openai.py and adapters/anthropic.py. Exempted with justification per
# audit remediation step 1 (Phase 5-8 changelog backfill).
EXEMPT_FILES = {
    os.path.join("src", "gateway", "devmode.py"),
}

violations = []
for dirpath, dirnames, filenames in os.walk(root):
    if "adapters" in dirpath:
        continue
    for f in filenames:
        if not f.endswith(".py"):
            continue
        path = os.path.join(dirpath, f)
        norm = os.path.normpath(path)
        if norm in ALLOWLIST or norm in EXEMPT_FILES:
            continue
        with open(path, encoding="utf-8") as fh:
            raw_lines = fh.readlines()
        # Strip comments so prose like 'no "anthropic", "openai"...' isn't flagged.
        code_lines = [ln.split("#", 1)[0] for ln in raw_lines]
        # Remove triple-quoted blocks (docstrings) from the code text.
        text = re.sub(r'""".*?"""', "", "".join(code_lines), flags=re.DOTALL)
        text = re.sub(r"'''.*?'''", "", text, flags=re.DOTALL)
        for lineno, line in enumerate(text.splitlines(), 1):
            if VENDOR_NAME.search(line) or COMPARISON.search(line):
                violations.append(f"{path}:{lineno}: {line.rstrip()}")

if violations:
    print("ARCHITECTURE VIOLATIONS:")
    for v in violations:
        print(v)
    # Fixes audit F-06: the gate previously printed violations but exited 0,
    # making CI decorative. A gate that cannot fail is not a gate.
    sys.exit(1)
else:
    print("CLEAN: No architecture violations found outside adapters/")
    sys.exit(0)

