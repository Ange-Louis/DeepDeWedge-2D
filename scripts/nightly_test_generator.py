import ast
from pathlib import Path

class TestGenerator:
    def __init__(self):
        self.repo_root = Path.cwd()
        self.src_dir = self.repo_root / "src" / "ddw"
        self.test_dir = self.repo_root / "tests"
    
    def get_all_source_functions(self):
        modules = {}
        if not self.src_dir.exists():
            print("Warning: src/ddw/ directory not found")
            return modules
        for py_file in self.src_dir.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue
            rel_path = py_file.relative_to(self.src_dir)
            module_path = "src.ddw." + str(rel_path).replace("/", ".").replace(".py", "")
            try:
                with open(py_file, 'r') as f:
                    tree = ast.parse(f.read())
                functions = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                        functions.append(node.name)
                if functions:
                    modules[module_path] = functions
            except Exception as e:
                print(f"Warning: {py_file}: {e}")
        return modules
    
    def get_existing_tests(self):
        tested_modules = set()
        if not self.test_dir.exists():
            return tested_modules
        for test_file in self.test_dir.rglob("test_*.py"):
            try:
                with open(test_file, 'r') as f:
                    content = f.read()
                for line in content.split("\n"):
                    if "from src.ddw" in line:
                        tested_modules.add(line.split("import")[0].strip())
            except Exception:
                pass
        return tested_modules
    
    def generate_test_file(self, module_path, functions):
        rel_path = module_path.replace("src.ddw.", "").replace(".", "/")
        test_rel_path = "test_" + rel_path
        test_file = self.test_dir / (test_rel_path + ".py")
        test_file.parent.mkdir(parents=True, exist_ok=True)
        import_stmt = "from " + module_path + " import " + ", ".join(functions)
        test_classes = []
        for func in functions:
            test_class = "\nclass Test" + func.capitalize() + ":\n    def test_" + func + "_basic(self):\n        pass\n"
            test_classes.append(test_class)
        test_code = "import pytest\n" + import_stmt + "\n\n" + "".join(test_classes)
        with open(test_file, 'w') as f:
            f.write(test_code)
        return test_file
    
    def run(self):
        print("Scanning source code in src/ddw/...")
        source_modules = self.get_all_source_functions()
        existing_tests = self.get_existing_tests()
        generated_files = []
        for module, functions in source_modules.items():
            if module not in existing_tests:
                test_file = self.generate_test_file(module, functions)
                generated_files.append(test_file)
                print(f"Generated: {test_file.relative_to(self.repo_root)}")
        if generated_files:
            print(f"\nGenerated {len(generated_files)} test files")
        else:
            print("\nNo new tests needed")
        return generated_files

if __name__ == "__main__":
    generator = TestGenerator()
    generator.run()
