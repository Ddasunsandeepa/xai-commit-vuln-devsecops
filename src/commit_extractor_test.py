from pydriller import Repository

repo_url = "https://github.com/gitpython-developers/GitPython"

for commit in Repository(repo_url).traverse_commits():
    print("Commit Hash:", commit.hash)
    print("Message:", commit.msg)
    print("Modified Files:", len(commit.modified_files))
    print("-" * 50)
