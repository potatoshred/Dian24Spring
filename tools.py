import matplotlib.pyplot as plt


def Visualizer(losses, lr, epochs, title="Loss", xlabel="Epoch", ylabel="loss"):
    # 创建一个图表，x 轴是历元，y 轴是损失
    plt.figure()

    # 绘制训练损失曲线
    plt.plot(range(1, len(losses) + 1), losses, label=f"lr={lr} epochs={epochs}")

    # 添加图例
    plt.legend()

    # 设置图表标题和标签
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 显示图表
    plt.show()

def MetricsVisualizer(arr, lr, epochs, title="Metrics", xlabel="Epoch", ylabel="Rate"):
    # 创建一个图表，x 轴是历元，y 轴是损失
    plt.figure()

    # 绘制训练损失曲线
    plt.plot(range(1, epochs + 1), arr[0], label=f"Accuracy")
    plt.plot(range(1, epochs + 1), arr[1], label=f"Precision")
    plt.plot(range(1, epochs + 1), arr[2], label=f"Recall")
    plt.plot(range(1, epochs + 1), arr[3], label=f"F1 Score")

    # 添加图例
    plt.legend()

    # 设置图表标题和标签
    plt.title(title+f" lr={lr}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 显示图表
    plt.show()