# Dry Bean Dataset 多分类机器学习期末项目

本项目是 AIT209 机器学习课程期末作业，围绕 Dry Bean Dataset 完成一个端到端机器学习工程流程，包括数据分析、数据清洗、特征工程、多算法实验、鲁棒性测试、工程化命令行入口和静态网页展示。

## 论文正文

飞书论文链接：

https://my.feishu.cn/wiki/XMs7wfRWcizbiskH0accxKSvn0d?from=from_copylink

## 项目内容

- 原始脏数据读取与质量分析
- 标签污染修复，例如 `D3RMAS0N`、`S3K3R`、`H0R0Z`、`B0MBAY`
- 缺失值、非法数值、负数异常处理
- 使用训练集中位数填补缺失值，避免测试集信息泄漏
- 使用训练集均值和标准差进行特征标准化
- 实现并比较三种多分类算法
- 分析测试集精度、Macro-F1、推理速度、loss 曲线、过拟合情况和鲁棒性
- 提供统一命令行入口和静态展示页面

## 运行方式

本项目使用 Python 3 标准库实现，无需额外安装第三方依赖。

```bash
python3 src/main.py clean
python3 src/main.py eda
python3 src/main.py train
python3 src/main.py robustness
```

运行完整流程：

```bash
python3 src/main.py all
```

本地展示页面：

```text
app/index.html
```

## 算法对比

| 模型 | 训练集准确率 | 验证集准确率 | 测试集准确率 | 测试集 Macro-F1 | 单样本推理耗时 |
|-|-:|-:|-:|-:|-:|
| Gaussian Naive Bayes | 0.896033 | 0.906459 | 0.902813 | 0.910464 | 0.021496 ms |
| Nearest Centroid | 0.882984 | 0.906459 | 0.898064 | 0.912241 | 0.011260 ms |
| Softmax Regression | 0.916763 | 0.912398 | 0.923639 | 0.935305 | 0.007531 ms |

在当前清洗策略和实验设置下，Softmax Regression 在测试集准确率和 Macro-F1 上表现最好。

## 鲁棒性实验

鲁棒性实验在训练数据中加入不同类型和不同强度的噪声，然后在干净测试集上评估模型表现。

| 噪声类型（强度 0.20） | Gaussian NB 准确率下降 | Nearest Centroid 准确率下降 | Softmax Regression 准确率下降 |
|-|-:|-:|-:|
| Gaussian feature noise | 0.006576 | 0.001462 | -0.001097 |
| Feature dropout | 0.013153 | 0.012788 | 0.018268 |
| Label flip | 0.017903 | 0.026672 | 0.086956 |

实验结果显示，标签翻转对 Softmax Regression 影响最大，说明监督标签质量对判别式模型非常重要。

## 目录结构

```text
final_project/
  app/
    index.html
  data/
    raw/
    processed/
  outputs/
    metrics/
    models/
    reports/
  src/
    common.py
    data_cleaning.py
    eda.py
    main.py
    ml_core.py
    plot_eval.py
    run_experiments.py
    run_robustness.py
  README.md
```

## 输出文件

- `data/processed/dry_bean_clean_train.csv`
- `data/processed/dry_bean_clean_val.csv`
- `data/processed/dry_bean_clean_test.csv`
- `outputs/reports/cleaning_report.json`
- `outputs/reports/eda_summary.json`
- `outputs/reports/baseline_results.json`
- `outputs/reports/robustness_results.json`
- `outputs/metrics/baseline_metrics.csv`
- `outputs/metrics/robustness_metrics.csv`
- `outputs/models/*.pkl`
- `app/index.html`

说明：图表文件可通过运行 `python3 src/main.py eda`、`python3 src/main.py train` 和 `python3 src/main.py robustness` 在本地生成。当前仓库未上传 `outputs/figures/` 和 `paper/` 文件夹。

## 关于 GitHub README 展示页面

GitHub README 不能直接嵌入并渲染 `app/index.html` 这个完整网页。可行方式有两种：

1. 使用 GitHub Pages 部署 `app/index.html`，然后在 README 中放访问链接。
2. 上传一张网页截图或 GIF，并在 README 中用图片形式展示页面效果。

如果需要完整展示页面中的图表，必须同时上传页面引用的图片资源，或者重新调整页面资源路径。
