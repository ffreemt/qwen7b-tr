# qwen7b-tr
[![pytest](https://github.com/ffreemt/qwen7b-tr/actions/workflows/routine-tests.yml/badge.svg)](https://github.com/ffreemt/qwen7b-tr/actions)[![python](https://img.shields.io/static/v1?label=python+&message=3.8%2B&color=blue)](https://www.python.org/downloads/)[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)[![PyPI version](https://badge.fury.io/py/qwen7b-tr.svg)](https://badge.fury.io/py/qwen7b-tr)

Translate/Chat using a qwen-7b-chat API at huggingface

## Install it

```shell
pip install qwen7b-tr --upgrade
# pip install git+https://github.com/ffreemt/qwen7b-tr
# poetry add git+https://github.com/ffreemt/qwen7b-tr
# git clone https://github.com/ffreemt/qwen7b-tr && cd qwen7b-tr
```

## Use it
```python
from qwen7b_tr.__main__ import qwen7b_tr

# This is in fact just chat with the qwen-7b-chat model
print(qwen7b_tr("你好"))
# 你好！有什么我能帮助你的吗？
```

## From command line
```
python -m qwen7b_tr test abc
# or qwen7b-tr test abc
```
```bash
三个版本如下：

1. 测试abc

2. ABC测试

3. 三号测试 ABC
```
If no text is provided, the content of the clipboard will be used.
```
python -m qwen7b_tr
```
```bash
# Assume the clipboard contains `available`
        No text provided, translating the content of the clipboard...
        diggin...
Loaded as API: https://mikeee-qwen-7b-chat.hf.space/ ✔

available

以下是三个不同的中文版本：

1. 可用的

2. 可获得的

3. 有效的
```
Target language can be specified with -t
```
python -m qwen7b_tr -t 德语
```
```
        No text provided, translating the content of the clipboard...
        diggin...
Loaded as API: https://mikeee-qwen-7b-chat.hf.space/ ✔

available

1. "Verfügbare"
  2. "Sollte"
  3. "Da ist"
```
```
python -m qwen7b_tr -t 英语 我国服务贸易“朋友圈”日益扩大
        diggin...
Loaded as API: https://mikeee-qwen-7b-chat.hf.space/ ✔

我国服务贸易“朋友圈”日益扩大

1. China's circle of service trade friends is expanding.
  2. The circle of China's service trade partners is growing larger.
  3. China's circle of service trade acquaintances is increasing.
```

## Help and Manual
```
python -m qwen7b_tr --help

# or
qwen7b-tr --help
```

```bash
Usage: python -m qwen7b_tr [OPTIONS] [QUESTION]...

  Translate via qwen-7b-chat huggingface API.

Arguments:
  [QUESTION]...  Source text or question.

Options:
  -c, --clipb                     Use clipboard content if set or if
                                  `question` is empty.
  -t, --to-lang TEXT              Target language when using the default
                                  prompt. [default: 中文]
  -n, --numb INTEGER              number of translation variants when using
                                  the default prompt. [default 3]
  -m, --max-new-tokens INTEGER    Max new tokens. [default: 256]
  --temperature, --temp INTEGER   Temperature. [default: 0.81]
  --repetition-penalty, --rep FLOAT
                                  Repetition penalty. [default: 1.1]
  --top-k, --top_k INTEGER        Top_k. [default: 0]
  --top-p, --top_p FLOAT          Top_p. [default: 0.9]
  --user-prompt TEXT              User prompt. [default: '翻成中文，列出3个版本.']
  -p, --system-prompt TEXT        User defined system prompt. [default: 'You
                                  are a helpful assistant.']
  -v, -V, --version               Show version info and exit.
  --help                          Show this message and exit.
```

## More Examples
### temperature
```
# default temperature 0.81
python -m qwen7b_tr marketing is critical to ensure fan interest remains high

1. 营销对于确保粉丝的兴趣保持高水平至关重要。
2. 市场营销对于维持粉丝的兴趣至关重要。
3. 粉丝的兴趣水平需要通过有效的市场营销来保持。
```

```
# a high temperature reuslts in a versatile outcome
python -m qwen7b_tr --temp 1.1 marketing is critical to ensure fan interest remains high

1. 营销至关重要，以确保球迷的兴趣继续保持高水平。
2. 营销工作是至关重要的，以保持球迷的兴趣。
3. 推广是保持球迷关注度的关键。
```
A very high temperature (for example 1.5) may result in  some nonsensical output.

### `-n`: number of translation variants
If you wish to have more choices, for example 5 variants
```
python -m qwen7b_tr -n 5 marketing campaign companion

  1. 营销活动伙伴
  2. 市场营销活动伴侣
  3. 营销活动助手
  4. 市场营销活动支持者
  5. 营销活动协作者

```
### max new tokens

The default `max_new_tokens` is 256, sutiable for chatting. If you translate a large chunk of text. you may wish to set `max_new_tokens` to 1024.
```
python -m qwen7b_tr --max-new-tokens 1024 blah blah ...
```

## Develop and Debug
```
set LOGURU_LEVEL=TRACE

# or in linux
export LOGURU_LEVEL=TRACE
```
to see a lot of debug messages