# 作业 1 中文译稿

- `hw1_zh.tex`：XeLaTeX 与 `ctex` 排版源文件。
- `hw1_zh.pdf`：编译后的中文译稿。
- 翻译模型：gpt-6-luna。题目内容仅作翻译呈现，没有解答作业或运行实验。

在工作目录中编译两次以更新交叉引用：

```bash
xelatex -interaction=nonstopmode -output-directory=translation/build translation/hw1_zh.tex
xelatex -interaction=nonstopmode -output-directory=translation/build translation/hw1_zh.tex
cp translation/build/hw1_zh.pdf translation/hw1_zh.pdf
```
