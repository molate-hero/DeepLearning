# 作业报告

`report.tex` 是报告源文件，`report.pdf` 是编译后的可阅读版本。

- `question1/01.pdf` 至 `04.pdf` 对应第一部分的四个小题，共 13 页。原文件保持不变，报告逐页插入，并在扫描页边缘叠加 0.3 pt 黑色细框。
- 报告使用 A4 纸，四周页边距均为 18 mm。
- `figures/` 中的 PDF 是报告使用的图表；PNG 仅用于本地预览，不提交。
- 第二部分的表格来自 `../HW1_Release/results/numpy.json`，损失图使用同目录下的 `numpy_loss.pdf`。
- 第三部分的训练和诊断图来自 `language.json`；探索实验来自 `gate_weights_seed416.json`。

## 编译

需要 XeLaTeX、latexmk，以及 ctex、graphicx、geometry、float、adjustbox 宏包。

在本目录运行：

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build report.tex
cp build/report.pdf report.pdf
```

编译中间文件位于 `build/`，不提交到 Git。

## 重新绘图

在仓库根目录运行：

```bash
python scripts/plot_training_curves.py
python scripts/plot_routing_diagnostics.py
python scripts/explore_gate_weights.py --plot-only
```

若需重新评估探索实验，运行 `python scripts/explore_gate_weights.py --device cuda`；该命令只评估已有检查点，不重新训练，也不覆盖 `language.json`。
