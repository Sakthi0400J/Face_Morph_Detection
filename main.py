import subprocess
import sys
import os
import config


def run(script):
    print(f"\nRunning {script}")
    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"Error in {script}")
        sys.exit(1)


def main():

    print("\n===== FACE FORENSIC PIPELINE =====")

    # build database only once
    if not os.path.exists(config.EMBEDDINGS_PATH):
        run("store_database.py")
    else:
        print("Embeddings already exist — skipping creation.")

    run("filter.py")
    run("find_morphed.py")

    print("\n===== PIPELINE COMPLETE =====")


if __name__ == "__main__":
    main()