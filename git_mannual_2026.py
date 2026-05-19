"""
检查是否有brew
"""

from difflib import restore


brew --version
如果有版本号，说明已经安装。

git --version
如果有版本号，说明 git 已安装。



**** INSTALL git ********************************************************

----- MAC ------
# STEP1-- Install homebrew (if needed).

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew doctor

# STEP2 install git

brew install git

----- Window -----

# Download Git from Git for Windows (https://gitforwindows.org/) and install it.

----- Linux -----
# Debian-based linux systems
sudo apt update
sudo apt install -y git

# Red Hat-based linux systems
sudo dnf install -y git
# or
sudo yum install -y git




****** git STARTER ******************************************************

# first-time config (recommended)
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
git config --global init.defaultBranch main
git config --global --list

# Choose a local directory to clone a remote repository.

git clone ssh://git@gitlab.huoyfish.com:222/gaojiejun/diguods_wd.git

'''after this step, the repository has been cloned into your new directory '''

'''当出现如下错误时,
 『致命错误：目标路径 'pathxxx' 已经存在，并且不是一个空目录。』
处理方式通常是下面三种之一：'''
# 1) 换一个新目录重新 clone
# 2) 清空目录后再 clone
# 3) 把已有目录接入远程仓库
git init
git remote add origin <repo_url>
git fetch
git checkout -t origin/main




****** git MainCourse ******************************************************



# Add the change to stage area
git add .

# Commit to local HEAD, then push to remote.
git commit -m "your commit message"

# Pull from remote repo first. (Technically fetch + rebase/merge remote changes first.)
git pull --rebase
# Push to remote origin.
git push
# first push for a new branch:
# git push -u origin <branch_name>


    ****** Branch  ******************************************************
    # Check current branches
    git branch
    # Create a new branch 'dev' and switch to it.
    git switch -c dev

    # Switch back to main
    git switch main

    # del a local branch
    git branch -d dev
    # force del local branch (careful)
    git branch -D dev

    # Push branch to remote repo
    git push -u origin <branch_name>
    # del remote branch
    git push origin --delete <branch_name>


    ****** Merge  ******************************************************

    # Merge branch dev to your current branch, say, main
    git switch main
    git pull --rebase
    git merge dev
    # conflict between two branches leads to automatic merge failing;
    # fix conflicts and then commit the result (add and commit).

    # You also can check the difference.
    git diff <currentBranch> <mergeBranch>


****** 恢复文件/提交  ******************************************************

git restore  某个文件  # 恢复某个文件到上一个版本
git restore .   # 恢复所有文件到上一个版本

如果文件已经 add 了怎么办
git restore --staged a.txt  # 恢复某个文件到暂存区
git restore --staged .  # 恢复所有文件到暂存区

如果文件已经 commit 了怎么办
git log --oneline  # 查看commit历史 

    git reset HEAD a.txt  # 恢复某个文件到上一个版本  【危险   建议使用revert  】
    git reset HEAD .  # 恢复所有文件到上一个版本
    git reset --hard HEAD  # 恢复所有文件到上一个版本
    git reset --hard HEAD^  # 恢复所有文件到上一个版本
    git reset --hard HEAD^^  # 恢复所有文件到上一个版本


回退后推送远程（重要）
git push -f


git revert 某个commit_id  # 恢复某个commit  【更安全】新增一个“反向提交”不会改历史。






****** ABOUT SSH  ******************************************************
# --- check ssh keys (recommended key type: ed25519)
ls ~/.ssh

# show public key and copy it to github account/settings
cat ~/.ssh/id_ed25519.pub
pbcopy < ~/.ssh/id_ed25519.pub

# If you don't have it, create one:
ssh-keygen -t ed25519 -C "your_email@example.com"

# add key to ssh-agent (to avoid repeated passphrase input)
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# test github ssh
# 成功会显示：
# Hi xxx! You've successfully authenticated...
ssh -T git@gitlab.huoyfish.com -p 222



# when git doesn't work and prompt:
>> fatal: bad config line 1 in file /Users/Username/.gitconfig
# solution:
# back up broken config, then re-create by git config commands
mv ~/.gitconfig ~/.gitconfig.bak
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"

'''
-------------
# @References:
-------------
https://git-scm.com/docs
https://rogerdudler.github.io/git-guide/
'''




*******常见命令行   *****************************************************

pwd # 查看当前目录
ls # 查看当前目录下的文件
    ls -l # 查看当前目录下的文件详细信息
    ls -a # 查看当前目录下的所有文件，包括隐藏文件
    ls -h # 查看当前目录下的文件大小
    ls -lh # 查看当前目录下的文件大小，以人类可读的方式显示
    ls -la # 查看当前目录下的所有文件，包括隐藏文件，以人类可读的方式显示
    ls -lh # 查看当前目录下的文件大小，以人类可读的方式显示
    ls -lh # 查看当前目录下的文件大小，以人类可读的方式显示
cd # 切换目录
cd .. # 切换到上一级目录
cd ~ # 切换到用户主目录
cd / # 切换到根目录
cd - # 切换到上次目录
cd ~user # 切换到用户主目录

mkdir # 创建目录
    mkdir -p # 创建目录，如果目录不存在，则创建
    mkdir -p a/b/c # 创建目录，如果目录不存在，则创建

rm # 删除文件
    rm -f # 删除文件，不提示
    rm -rf # 删除目录，不提示
    rm -rf * # 删除当前目录下的所有文件
    rm -rf . # 删除当前目录下的所有文件
    rm -rf .. # 删除当前目录下的所有文件
    rm -rf ... # 删除当前目录下的所有文件
    rm -rf .... # 删除当前目录下的所有文件
    rm -rf ..... # 删除当前目录下的所有文件


clear # 清屏

mv # 移动文件
    mv -i 文件名或文件夹（可以写多个） 目标路径 # 移动文件，如果目标路径存在，则提示
cp # 复制文件

