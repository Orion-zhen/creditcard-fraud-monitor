import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--method", "-m", default="dl", help="选择训练方式，深度学习(dl)或传统的机器学习(ml)，选择ml时，其余参数均不可用", choices=["dl", "ml"])
parser.add_argument("--decimation", "-de", action="store_true", default=True, help="是否对数据集使用降采样")
parser.add_argument("--epochs", "-e", default=500, type=int, help="训练的轮数")
parser.add_argument("--lr", "-l",default=0.00001, type=float, help="学习率")
parser.add_argument("--batch-size", "-b",default=16, type=int, help="batch size")
parser.add_argument("--split", "-s", default=0.8, type=float, help="训练集占比")
parser.add_argument("--device", "-to", default="cuda", help="训练时使用的设备，其中xpu代表intel显卡，ipex代表intel CPU", choices=["cpu", "cuda", "xpu", "ipex"])
parser.add_argument("--output-path", "-o", default="./output", help="输出文件夹")


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)