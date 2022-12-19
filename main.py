import numpy as np
import matplotlib.pyplot as plt

'''试验transpose()
def back (a,b):
    return a,b

if __name__ == '__main__':
    a = np.array([[1,2,3],[11,12,13],[21,22,23]])
    print(a)
    b = np.array([[31,32,33],[41,42,43],[51,52,53]])
    print(b)
    a, b = transpose(back(a,b))
    #a, b = back(a, b)
    print(a)
    print(b)
'''


# 数据加载器基类
class Loader(object):
    def __init__(self, path, count):
        '''
        初始化加载器
        path: 数据文件路径
        count: 文件中的样本个数
        '''
        self.path = path
        self.count = count

    def get_file_content(self):
        '''
        读取文件内容
        '''
        f = open(self.path, 'rb')
        content = f.read()
        f.close()
        return content

    def to_int(self, byte):
        '''
        将unsigned byte字符转换为整数
        '''
        # print(byte)
        # return struct.unpack('B', byte)[0]
        return byte


# 图像数据加载器
class ImageLoader(Loader):
    def get_picture(self, content, index):
        '''
        内部函数，从文件中获取图像
        '''
        start = index * 28 * 28 + 16
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(
                    self.to_int(content[start + i * 28 + j]))
        return picture

    def get_one_sample(self, picture):
        '''
        内部函数，将图像转化为样本的输入向量
        '''
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    def load(self):
        '''
        加载数据文件，获得全部样本的输入向量
        '''
        content = self.get_file_content()
        data_set = []
        for index in range(self.count):
            data_set.append(
                self.get_one_sample(
                    self.get_picture(content, index)))
        return data_set


# 标签数据加载器
class LabelLoader(Loader):
    def load(self):
        '''
        加载数据文件，获得全部样本的标签向量
        '''
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index + 8]))
        return labels

    def norm(self, label):
        '''
        内部函数，将一个值转换为10维标签向量
        '''
        label_vec = []
        label_value = self.to_int(label)
        for i in range(10):
            if i == label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec


def get_training_data_set():
    '''
    获得训练数据集
    '''
    image_loader = ImageLoader('/Users/huangzikun/Desktop/TTTC6404 Image Processing and Computer '
                               'Vision/project/dataset/train-images-idx3-ubyte', 60000)
    label_loader = LabelLoader('/Users/huangzikun/Desktop/TTTC6404 Image Processing and Computer '
                               'Vision/project/dataset/train-labels-idx1-ubyte', 60000)
    return image_loader.load(), label_loader.load()


def get_test_data_set():
    '''
    获得测试数据集
    '''
    image_loader = ImageLoader('/Users/huangzikun/Desktop/TTTC6404 Image Processing and Computer Vision/project/dataset/t10k-images-idx3-ubyte', 10000)
    label_loader = LabelLoader('/Users/huangzikun/Desktop/TTTC6404 Image Processing and Computer Vision/project/dataset/t10k-labels-idx1-ubyte', 10000)
    return image_loader.load(), label_loader.load()


if __name__ == '__main__':
    train_data_set, train_labels = get_training_data_set()
    line = np.array(train_data_set[0])
    img = line.reshape((28, 28))
    plt.imshow(img)
    plt.show()