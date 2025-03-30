import ast
import os
import pkg_resources
import sys
from tabulate import tabulate

def get_imported_modules(file_path):
    """Trích xuất danh sách thư viện được import trong file Python."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=file_path)
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"⚠️ Lỗi trong file {file_path}: {e}")
        return set()

    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split('.')[0])

    return imported_modules

def find_python_files(root_dir):
    """Tìm tất cả các file .py trong thư mục gốc và thư mục con."""
    python_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".py"):
                python_files.append(os.path.join(dirpath, file))
    return python_files

def get_installed_version(module_name):
    """Lấy phiên bản của module nếu nó đã được cài (loại bỏ module built-in)."""
    if module_name in sys.builtin_module_names:
        return None  # Bỏ qua module built-in
    try:
        return pkg_resources.get_distribution(module_name).version
    except pkg_resources.DistributionNotFound:
        return "Not Installed"

def main():
    root_dir = os.getcwd()
    python_files = find_python_files(root_dir)

    all_imports = set()
    for file in python_files:
        all_imports.update(get_imported_modules(file))

    # Lọc thư viện từ PyPI
    library_versions = {
        lib: get_installed_version(lib) for lib in sorted(all_imports)
        if get_installed_version(lib) is not None
    }

    # Chuyển dữ liệu thành bảng
    table_data = [[lib, version] for lib, version in library_versions.items()]
    
    print("\n📌 Thư viện cài đặt từ PyPI được sử dụng trong project:")
    print(tabulate(table_data, headers=["Thư viện", "Phiên bản"], tablefmt="grid"))

if __name__ == "__main__":
    main()
