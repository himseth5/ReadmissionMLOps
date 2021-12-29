#! /usr/bin/bash
git clone https://github.com/himseth5/ReadmissionMLOps-new.git
cd ReadmissionMLOps-new/
git remote -v
git remote add upstream https://github.com/himseth5/ReadmissionMLOps.git
git remote -v
git fetch upstream
git checkout main
git merge upstream/master
git push origin
