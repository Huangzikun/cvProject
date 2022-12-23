import numpy
import numpy as np
from copy import copy
import DatasetLoader

minLeafNum = 10


# 5 0 1 2
# ->
# 0 0 0 0 1 0 0 0 0
# 1 0 0 0 0 0 0 0 0
# 0 1 0 0 0 0 0 0 0
# 0 0 1 0 0 0 0 0 0
def labels2Map(labels):
    classUnique = np.unique(labels)
    classNum = len(classUnique)
    classMap = np.zeros([len(labels), classNum])
    for i in range(len(labels)):
        classMap[i][labels[i]] = 1  # every point set 1
    return classMap, classUnique


def buildTree(datas, labels, features):
    counts, oneCount = datas.shape

    # 如果分清楚了，就结束
    if len(np.unique(labels)) == 1:
        return labels[0]

    # 若训练集的所有特征都被使用完毕，当前无可用特征，但样本仍未被分“纯”
    # 则返回所含样本最多的标签作为结果
    if counts == 1:
        return np.bincount(labels).argmax()

    bestIndexFeature, bestSpiltPoint = chooseBestFeature(datas, labels, features)
    # 得到最佳特征
    bestFeature = features[bestIndexFeature]
    # 初始化决策树
    decisionTree = {bestFeature: {}}
    # 使用过当前最佳特征后将其删去
    del features[bestIndexFeature]
    # 子特征 = 当前特征（因为刚才已经删去了用过的特征）
    # sub_labels = features[:]

    subData, subLabels, subDataOther, subLabelsOther = splitData(datas, labels, bestIndexFeature, bestSpiltPoint)
    decisionTree[bestFeature][bestSpiltPoint] = buildTree(np.array(subData), subLabels, features[:])
    decisionTree[bestFeature]['other'] = buildTree(np.array(subDataOther), subLabelsOther, features[:])

    return decisionTree


def calGini(datas, labels):
    labelCnt = {}
    if len(labels) == 0:
        return 1

    binaryCount = np.bincount(np.array(labels))

    # 得到了当前集合中每个标签的样本个数后，计算它们的p值
    for i in range(len(binaryCount)):
        labelCnt[i] = binaryCount[i] / float(len(datas))
        labelCnt[i] = labelCnt[i] * labelCnt[i]
    # 计算Gini系数
    Gini = 1 - sum(labelCnt.values())
    return Gini


# 寻找最佳切割点
def chooseBestFeature(datas, labels, features):
    bincountLabels = np.bincount(labels)
    # 初始化最佳基尼系数
    bestGini = 1
    # 初始化最优特征
    bestIndexFeature = -1
    bestSpiltPoint = -1

    # 循环28*28
    for i in range(len(features)):
        # 每个点的值唯一
        uniqueVals = set(datas[:, i])
        Gini = {}
        for value in uniqueVals:
            subData, subLabels, subDataOther, subLabelsOther = splitData(datas, labels, i, value)
            # 转成float
            proportionSubData = len(subData) / float(len(datas))
            proportionSubDataOther = len(subDataOther) / float(len(datas))
            giniSubData = calGini(subData, subLabels)
            giniSubDataOther = calGini(subDataOther, subLabelsOther)

            # 计算由当前最优切分点划分后的最终Gini系数
            Gini[value] = proportionSubData * giniSubData + proportionSubDataOther * giniSubDataOther
            # 更新最优特征和最优切分点
            if Gini[value] < bestGini:
                bestGini = Gini[value]
                bestIndexFeature = i
                bestSpiltPoint = value

    return bestIndexFeature, bestSpiltPoint


# 将当前样本集分割成特征i取值为value的一部分和取值不为value的一部分（二分）
def splitData(datas, labels, index, value):
    subData = []
    subLabels = []
    subDataOther = []
    subLabelsOther = []
    for i in range(len(datas)):
        data = datas[i]
        if data[index] == value:
            subData.append(data)
            subLabels.append(labels[i])
        else:
            subDataOther.append(data)
            subLabelsOther.append(labels[i])

    return subData, subLabels, subDataOther, subLabelsOther


def useTree(tree, inFeatures, testData):
    # 根节点代表的属性
    label = None

    firstFeature = list(tree.keys())[0]

    other = tree[firstFeature]

    indexFirstFeature = inFeatures.index(firstFeature)

    for key in other:
        if key != 'other':
            if testData[indexFirstFeature] == key:
                if isinstance(other[key], dict):
                    label = useTree(other[key], inFeatures, testData)
                else:
                    label = other[key]
            else:
                if isinstance(other['other'], dict):
                    label = useTree(other['other'], inFeatures, testData)
                else:
                    label = other['other']

    return label


def main():
    [dataX, dataY] = DatasetLoader.getTrainingDataSet()
    testX = []
    for i in range(1000):
        testX.append(np.array(dataX[i]))
    x = np.array(testX)

    varNameList = []
    for i in range(len(dataX[0])):
        varNameList.append(i)

    textY = []
    for i in range(1000):
        textY.append(dataY[i])
    y = np.array(textY)

    features = list(range(28 * 28))
    tree = buildTree(x, y, features)
    print(tree)

    [testData, testLabels] = DatasetLoader.getTrainingDataSet()
    testX = []
    for i in range(10):
        testX.append(np.array(testData[i]))
    x = np.array(testX)

    tLabels = []
    for i in range(10):
        tLabels.append(testLabels[i])

    t = np.array(tLabels)

    print(t)

    for t in x:
        label = useTree(tree, list(range(28*28)), t)
        print("tree: " + str(label))



if __name__ == '__main__':
    main()
