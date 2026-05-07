from pydriller import Repository
import csv

repo_url = "https://github.com/gitpython-developers/GitPython"

output_file = "data/commit_dataset.csv"

count = 0

with open(output_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    # CSV Header
    writer.writerow([
        "commit_hash",
        "commit_message",
        "files_changed",
        "diff"
    ])

    for commit in Repository(repo_url).traverse_commits():

        full_diff = ""

        for modified_file in commit.modified_files:

            if modified_file.diff:
                full_diff += modified_file.diff + "\n"

        writer.writerow([
            commit.hash,
            commit.msg,
            len(commit.modified_files),
            full_diff
        ])

        count += 1

        print(f"Processed commit {count}: {commit.hash}")

        # Limit for testing
        if count == 20:
            break

print("Commit extraction completed.")