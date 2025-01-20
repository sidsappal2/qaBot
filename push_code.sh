#!/bin/bash

# Prompt for branch
echo "Please enter your branch name: "
read $branch_name

# Ensure you are on the 'new' branch
#git checkout new

# Add all changes to staging
git add .

# Prompt for commit message
echo "Please enter your commit message: "
read commit_message

# Commit the changes with the entered commit message
git commit -m "$commit_message"

# Push changes to the given branch on the remote repository
git push origin $branch_name
