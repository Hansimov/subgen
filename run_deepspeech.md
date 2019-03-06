<!-- 
DeepSpeech on Windows WSL
    https://fotidim.com/deepspeech-on-windows-wsl-287cb27557d4
 -->

**1. 安装 WSL**

`PowerShell` 运行：

```
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
```

或者 "控制面板 > 启用或关闭 Windows 功能 > 勾选适用于 Linux 的 Windows 子系统"。

重启电脑。

开始菜单输入 "Microsoft Store"，搜索 "Ubuntu"，选择安装那个没有版本号的。

安装完成后，第一次启动会要求输入用户名和密码。


```
deb http://mirrors.aliyun.com/ubuntu/ xenial main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ xenial-security main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ xenial-updates main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu/ xenial-backports main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ xenial main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-security main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-updates main restricted universe multiverse
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-backports main restricted universe multiverse
```

```
mkdir DeepSpeech; cd DeepSpeech
```

```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
```

```
sudo apt-get install python2.7
```

```
sudo ln -s /usr/bin/python3 /etc/alternatives/python
sudo ln -s /etc/alternatives/python /usr/bin/python
```

```
sudo apt-get install python-pip
```

```
git clone https://github.com/mozilla/DeepSpeech .
```

```
mkdir -p native_client/bin
pip install six
```

```
python util/taskcluster.py --target native_client/bin
```

```
cd native_client
```



```
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.4.1/audio-0.4.1.tar.gz
tar -xvzf audio-0.4.1.tar.gz
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.4.1/deepspeech-0.4.1-models.tar.gz
tar -xvzf deepspeech-0.4.1-models.tar.gz
```


```
https_proxy = http://127.0.0.1:1080/
http_proxy = http://127.0.0.1:1080/
ftp_proxy = http://127.0.0.1:1080/

use_proxy = on
```

```
pip install deepspeech --user
```

```
sudo apt-get install sox libsox-fmt-mp3
```

<!-- 
    deepspeech [-h] --model MODEL --alphabet ALPHABET [--lm [LM]] [--trie [TRIE]] --audio AUDIO [--version]
 -->

deepspeech --model ../models/output_graph.pb --alphabet ../models/alphabet.txt --lm ../models/lm.binary --trie ../models/trie --audio /mnt/f/Sources/git/subgen/tests/part_43.wav


2830-3980-0043.wav
4507-16021-0012.wav
8455-210777-0068.wav
/mnt/f/Sources/git/subgen/tests/ english.wav | eng_m4.wav | part_43.wav