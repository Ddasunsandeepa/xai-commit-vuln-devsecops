from pydriller import Repository

repo_url = "https://github.com/gitpython-developers/GitPython"

count = 0

for commit in Repository(repo_url).traverse_commits():

    print("=" * 80)
    print("Commit Hash:", commit.hash)
    print("Message:", commit.msg)

    for file in commit.modified_files:

        print("\nFile Name:", file.filename)

        if file.diff:
            print(file.diff[:1000])  # first 1000 chars only

    count += 1

    if count == 2:
        break