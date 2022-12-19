import numpy as np
import matplotlib.pyplot as plt


# 数据加载器基类
class DatasetLoader(object):
    def __init__(self, path, count):
        """
        初始化加载器
        path: 数据文件路径
        count: 文件中的样本个数
        """
        self.path = path
        self.count = count

    def getFileContent(self):
        """
        读取文件内容
        """
        f = open(self.path, 'rb')
        content = f.read()
        f.close()
        return content


# 图像数据加载器
def getPicture(content, index):
    """
    内部函数，从文件中获取图像
    """
    start = index * 28 * 28 + 16
    picture = []
    for i in range(28):
        picture.append([])
        for j in range(28):
            picture[i].append(
                content[start + i * 28 + j])
    return picture


def getOneSample(picture):
    """
    内部函数，将图像转化为样本的输入向量
    """
    sample = []
    for i in range(28):
        for j in range(28):
            sample.append(picture[i][j])
    return sample


class ImageLoader(DatasetLoader):

    def load(self):
        """
        加载数据文件，获得全部样本的输入向量
        """
        content = self.getFileContent()
        data_set = []
        for index in range(self.count):
            data_set.append(
                getOneSample(
                    getPicture(content, index)))
        return data_set


# 标签数据加载器
def norm(label):
    """
    内部函数，将一个值转换为10维标签向量
    """
    label_vec = []
    label_value = label
    for i in range(10):
        if i == label_value:
            label_vec.append(0.9)
        else:
            label_vec.append(0.1)
    return label_vec


class LabelLoader(DatasetLoader):
    def load(self):
        """
        加载数据文件，获得全部样本的标签向量
        """
        content = self.getFileContent()
        labels = []
        for index in range(self.count):
            labels.append(norm(content[index + 8]))
        return labels


def getTrainingDataSet():
    """
    获得训练数据集
    """
    imageLoader = ImageLoader('/Users/huangzikun/Desktop/TTTC6404 Image Processing and Computer '
                               'Vision/project/dataset/train-images-idx3-ubyte', 60000)
    labelLoader = LabelLoader('/Users/huangzikun/Desktop/TTTC6404 Image Processing and Computer '
                               'Vision/project/dataset/train-labels-idx1-ubyte', 60000)
    return imageLoader.load(), labelLoader.load()


def getTestDataSet():
    """
    获得测试数据集
    """
    imageLoader = ImageLoader(
        '/Users/huangzikun/Desktop/TTTC6404 Image Processing and Computer Vision/project/dataset/t10k-images-idx3-ubyte',
        10000)
    labelLoader = LabelLoader(
        '/Users/huangzikun/Desktop/TTTC6404 Image Processing and Computer Vision/project/dataset/t10k-labels-idx1-ubyte',
        10000)
    return imageLoader.load(), labelLoader.load()


if __name__ == '__main__':
    trainDataSet, trainLabels = getTrainingDataSet()
    line = np.array(trainDataSet[0])
    img = line.reshape((28, 28))
    plt.imshow(img)
    plt.show()
