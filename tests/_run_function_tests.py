import runpy


count = 0
for path in ("tests/test_experiments.py", "tests/test_profiler.py", "tests/test_drift.py"):
    namespace = runpy.run_path(path)
    for name, test in namespace.items():
        if name.startswith("test_") and callable(test):
            test()
            count += 1
print(f"function tests passed: {count}")
