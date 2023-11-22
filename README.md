# creditcard-fraud-monitor

本项目基于[第 13 代英特尔® 酷睿™ i5 处理器 i5-13600K](https://www.intel.cn/content/www/cn/zh/products/sku/230493/intel-core-i513600k-processor-24m-cache-up-to-5-10-ghz/specifications.html)和[英特尔锐炫™ a 系列显卡 A770](https://www.intel.cn/content/www/cn/zh/products/sku/229151/intel-arc-a770-graphics-16gb/specifications.html)编写

在项目中使用了[Intel® Distribution of Modin*](https://www.intel.cn/content/www/cn/zh/developer/tools/oneapi/distribution-of-modin.html)、[Intel® Extension for PyTorch*](https://pytorch.org/tutorials/recipes/recipes/intel_extension_for_pytorch.html)和[Intel® Extension for Scikit-learn* ](https://www.intel.cn/content/www/cn/zh/developer/tools/oneapi/scikit-learn.html)对代码进行加速

使用深度神经网络构建分类器, 可以准确地识别信用卡诈骗情况, 准确率达到 **96.95%** !

## 快速上手

将本仓库克隆到本地:

```shell
git clone https://github.com/Orion-zhen/creditcard-fraud-monitor.git
cd creditcard-fraud-monitor
```

安装依赖:

```shell
pip install -r requirements.txt
```

在[展示文件](./demo.ipynb)中，阅读我们的处理流程并运行演示程序，能帮助您更快速地理解项目

## 进一步探索我们的深度神经网络

如果希望采用intel显卡完成训练和推理, 请运行:

```shell
pip install -r requirements-xpu.txt
pip install torch==2.0.1a0 intel_extension_for_pytorch==2.0.110+xpu -f https://developer.intel.com/ipex-whl-stable-xpu
```

> 值得注意的是, intel_extension_for_pytorch目前仅有Linux平台可用, 具体情况请参考[ipex官方文档](https://github.com/intel/intel-extension-for-pytorch)

开始训练:

```shell
python main.py
```

得到结果!😀

![eg-ipex](assets/eg-ipex.png)

> 更多可选参数可运行`python main.py -h`查看, 如果想使用传统机器学习方法, 请浏览`classic_ml.ipynb`

## 英特尔®技术

本项目使用了[intel_extension_for_pytorch](https://github.com/intel/intel-extension-for-pytorch)来对pytorch训练和推理进行优化和加速

想要使用ipex优化模型训练, 只需要在代码中加入如下几行:

```python
import intel_extension_for_pytorch as ipex

optimizer = torch.optim.something
model.train()
model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)
model = torch.compile(model, backend="ipex") # 实验性功能, 可以获得更强效果
```

想要使用ipex优化模型推理, 则可以这样:

```python
import intel_extension_for_pytorch as ipex

model.eval()
model = ipex.optimize(model)
model = torch.compile(model, backend="ipex") # 实验性功能, 可以获得更强效果
```

以下是使用ipex优化加速的结果和使用cuda加速的结果对比:

ipex

![eg-ipex](assets/eg-ipex.png)

cuda

![eg-cuda](assets/eg-cuda.png)

可以看到, ipex带来了非常可观的准确率提升
