#! /usr/bin/bash
git config --global user.name "himseth5"
git config --global user.email "himseth5@gmail.com"
#git config --global user.password "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"
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
