# encoding: utf-8
"""
Author: 沙振宇
CreateTime: 2018-12-25
UpdateTime: 2019-12-13
Info: 利用用TF特征向量和sim hash指纹计算中文文本的相似度试验
"""
import json
import sys
import time
import re_test
from text_similarity_master.src.isSimilar import DocFeatLoader

# 引用text_similarity 用TF特征向量和sim hash指纹计算中文文本的相似度(https://github.com/zyymax/text-similarity)
sys.path.append("./text_similarity_master/src")
from simhash_imp import SimhashBuilder
from tokens import JiebaTokenizer
from features import FeatureBuilder
from Utils import cosine_distance_nonzero
from DictBuilder import WordDictBuilder

user_dict = "./user_dict/user_ner_dict.txt"
word_dict_path = "./data/word.dict"
flow_data_dir = './flowData'
stopwords_path = "./text_similarity_master/data/stopwords.txt"

# 所有文本数据
labelContents = []
# 预加载所有流程信息
originalFlowMap = {}
processFlowMap = {}

# 配置文件flowId列表
conFileList = []

# 读取文件里面数据
def getDate(filename):
    with open(filename,"r",encoding='utf-8') as fp:
        data = fp.read()
    return data

# 加载数据
def loadData(flowId):
    nodeMap = getDate(flow_data_dir+"/"+flowId)
    nodeMaps = json.loads(nodeMap)
    originalFlowMap[flowId] = nodeMaps

    # 调用数据预处理  结果放到 processFlowMap 字典内
    processLabelDataMap = processLabelData(nodeMaps)
    processFlowMap[flowId] = processLabelDataMap

# 节点条件标注数据预处理
def splitConditionLabelData(condition):
    normalNodeProcessList = []
    try:
        labelData = condition["labelData"]
        targetNodeId = condition["targetNodeId"]
        actionCode = condition["actionCode"]
        # 修改错误检测
        try:
            conditionId = condition["conditionId"]
        except Exception as e:
            conditionId = ""
        labelDataList = labelData.split("||")

        for i in range(len(labelDataList)):
            tmpMap = {}
            labelDate = labelDataList[i]
            tmpMap["targetNodeId"] = targetNodeId
            tmpMap["labelData"] = labelDate
            labelDataAfterProcess = re_test.run(labelDate)
            tmpMap["processLabelData"] = labelDataAfterProcess
            tmpMap["actionCode"] = actionCode
            tmpMap["conditionId"] = conditionId
            labelContents.append(labelDataAfterProcess)
            normalNodeProcessList.append(tmpMap)
    except Exception as e:
        print("splitConditionLabelData error:", str(e))
    return normalNodeProcessList

# 知识库标注数据预处理
def splitKnowledgeLabelData(nodeId, labelData, actionCode):
    KnowledgeNodeProcessList = []
    try:
        labelDataList = labelData.split("||")
        for i in range(len(labelDataList)):
            tmpMap = {}
            labelDate = labelDataList[i]
            tmpMap["targetNodeId"] = nodeId
            tmpMap["labelData"] = labelDate
            labelDataAfterProcess = re_test.run(labelDate)
            tmpMap["processLabelData"] = labelDataAfterProcess
            tmpMap["actionCode"] = actionCode
            tmpMap["conditionId"] = ""
            labelContents.append(labelDataAfterProcess)
            KnowledgeNodeProcessList.append(tmpMap)
    except Exception as e:
        print("splitKnowledgeLabelData:", str(e))
    return KnowledgeNodeProcessList

# 数据预处理
def processLabelData(nodeMap):
    processLabelDataMap = {}
    splitKnowledgeLabelDataList = []
    for oneNodeId in nodeMap:
        try:
            oneNode = nodeMap[oneNodeId]
            if "8000" not in oneNodeId:
                if "conditions" in oneNode.keys():
                    conditions = oneNode["conditions"]
                    tmpLabelDataList = []
                    for i in range(len(conditions)):
                        condition = conditions[i]
                        if condition["labelData"] == "":
                            continue
                        tmpLabelDataList = tmpLabelDataList + splitConditionLabelData(condition)
                    if len(tmpLabelDataList) > 0:
                        processLabelDataMap[oneNodeId] = tmpLabelDataList
            elif "labelData" in oneNode.keys():
                labelData = oneNode["labelData"]
                actionCode = oneNode["actionCode"]
                splitKnowledgeLabelDataList = splitKnowledgeLabelDataList + splitKnowledgeLabelData(oneNodeId,labelData,actionCode)
        except Exception as e:
            print("error nodeID %s,processLabelData::%s" % (oneNodeId, e))
    if len(splitKnowledgeLabelDataList) > 0:
        processLabelDataMap["80000"] = splitKnowledgeLabelDataList
    return processLabelDataMap

# 生成词典
def buildWords(jt, contentList):
    doc_tokens_list = []
    for i in range(len(contentList)):
        doc_tokens = jt.tokens(contentList[i])
        doc_tokens_list.extend(doc_tokens)
    wdb = WordDictBuilder(word_dict_path, tokenlist=doc_tokens_list)
    wdb.run()
    wdb.save(word_dict_path)

    wordList = []
    with open(word_dict_path) as ins:
        for line in ins.readlines():
            if line != '\n':
                wordList.append(line.split()[1])
    wordDict = {}
    for idx, ascword in enumerate(wordList):
        wordDict[ascword] = idx

    return wordList, wordDict

# 生成特征向量
def generateDocFeatureVector(processLabelDataMap, jt, fb, smb):
    print("generate doc feature vector...")
    contentFlListMap = {}
    try:
        for id, contentList in processLabelDataMap.items():
            if len(contentList) == 0:
                continue
            tmpContentList = []
            for i in range(len(contentList)):
                content = contentList[i]
                processLabelData = content["processLabelData"]
                doc_token = jt.tokens(processLabelData)
                doc_feat = fb.compute(doc_token)
                doc_fl = DocFeatLoader(smb, doc_feat)
                content["lableDataFeatureVector"] = doc_fl
                content["lableDataToken"] = doc_token
                tmpContentList.append(content)
            contentFlListMap[id] = tmpContentList
        print("generate doc feature vector end")
    except Exception as e:
        print("generateDocFeatureVector:", str(e))
    return contentFlListMap

# 预处理主文件
def preProcessingData(filename):
    loadData(filename)
    jt_time = time.time()
    global jt
    jt = JiebaTokenizer(stopwords_path, 'c')
    end_jt_time = time.time()
    print('JiebaTokenizer time: %s' % str(end_jt_time - jt_time))
    # 根据所有的标注数据做词向量模型 生成词典
    wordList, wordDict = buildWords(jt, labelContents)
    end_build_time = time.time()
    print('buildWords time: %s' % str(end_build_time - end_jt_time))
    # 生成特征向量
    global fb
    fb = FeatureBuilder(wordDict)
    end_fb_build_time = time.time()
    print('FeatureBuilder time: %s' % str(end_fb_build_time - end_build_time))
    # 生成指纹
    global smb
    smb = SimhashBuilder(wordList)
    end_smb_build_time = time.time()
    print('SimhashBuilder time: %s' % str(end_smb_build_time - end_fb_build_time))
    # 生成所有标注数据的特征向量
    for flowId, processLabelDataMap in processFlowMap.items():
        processFlowMap[flowId] = generateDocFeatureVector(processLabelDataMap, jt, fb, smb)
    end_docFV_time = time.time()
    print('generateDocFeatureVector time: %s' % str(end_docFV_time - end_smb_build_time))

#语义相似度计算
def textSimilarity(question, nodeMap, nodeId):
    try:
        text = re_test.run(question) # 通过正则 查找匹配数据
        doc_token = jt.tokens(text) # 预处理，分词
        doc_feat = fb.compute(doc_token)
        doc_fl = DocFeatLoader(smb, doc_feat) # 对象包含两个参数 # fingerprint   指纹分值 # feat_vec  包含元组的列表

        # 预处理后的配置文件
        contentFlListMap = nodeMap
        p_score_list = []
        if nodeId in contentFlListMap.keys():
            nodeFlList = contentFlListMap[nodeId]
            print("nodeFilist",nodeFlList)
            for i in range(len(nodeFlList)):
                p_score_dict={}
                dist = cosine_distance_nonzero(nodeFlList[i]["lableDataFeatureVector"].feat_vec, doc_fl.feat_vec, norm=False)
                p_score_dict["score"] = dist
                p_score_dict["labelData"] = nodeFlList[i]["labelData"]
                p_score_dict["targetNodeId"] = nodeFlList[i]["targetNodeId"]
                p_score_dict["conditionId"] = nodeFlList[i]["conditionId"]
                p_score_list.append(p_score_dict)
            p_score_list = sorted(p_score_list, key=lambda score : score["score"], reverse=True)

            print("Sorted：",p_score_list)

            Complete_MayBeL4 = []
            Complete_MayBeL4Score = []
            Complete_MayBeL4ID = []
            Complete_MayBeL4Max = 3
            for i, el in enumerate(p_score_list):
                p_label = p_score_list[i]["labelData"]
                p_score = p_score_list[i]["score"]
                p_conditionId = p_score_list[i]["conditionId"]
                if len(Complete_MayBeL4) < Complete_MayBeL4Max:
                    Complete_MayBeL4.append(p_label)
                    Complete_MayBeL4Score.append(p_score)
                    Complete_MayBeL4ID.append(p_conditionId)
                else:
                    break

            print("************************************")
            print("用户问题：", question)
            print("相似问(Max=%s)：%s"%(Complete_MayBeL4Max,Complete_MayBeL4))
            print("特征值(Max=%s)：%s"%(Complete_MayBeL4Max,Complete_MayBeL4Score))
            print("ID：",Complete_MayBeL4ID)
            return "", "", "", "", "", ""
    except Exception as e:
        print("************************************")
        print("Error textSimilarity：", str(e))
        print("************************************")
    return "", "", "", "", "", ""

if __name__ =="__main__":
    file_name = "test.json"                     # 1、准备测试数据
    preProcessingData(file_name)                # 2、预处理读到的数据
    print("originalFlowMap:",originalFlowMap)   # 打印原配置文件
    print("processFlowMap:",processFlowMap)     # 打印预处理的配置文件
    nodeMap = processFlowMap[file_name]         # 3、加载数据到Map中
    userQuestion = "你打错号码了"               # 4、输入用户问题
    textSimilarity(userQuestion, nodeMap, "1")  # 5、利用TF特征向量和sim hash指纹计算出 预处理的配置文件中的分值
