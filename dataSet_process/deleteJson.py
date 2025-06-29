

import os
import sys

def delete_json_files(root_dir):
    """
    在 root_dir 及其所有子目录中查找并删除 .json 文件。
    """
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.json'):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    count += 1
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
    return count

def main():
    total = delete_json_files('./image/data')
    print(f"\nDone. Total .json files deleted: {total}")

if __name__ == "__main__":
    main()
