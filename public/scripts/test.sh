#! /usr/bin/bash
git config --global user.name "himseth5"
git config --global user.email "himseth5@gmail.com"
git config --global user.password "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"
#https://himseth5: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@github.com/himseth5/ReadmissionMLOps.git
git config credential.helper store
ls -la
git config --list
git clone https://github.com/himseth5/ReadmissionMLOps-new.git
cd ReadmissionMLOps-new/
git remote -v
git remote add upstream https://github.com/himseth5/ReadmissionMLOps.git
git remote -v
git fetch upstream
git checkout main
git merge upstream/master
git push https://himseth5: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@github.com/himseth5/ReadmissionMLOps-new.git
echo "Done"
