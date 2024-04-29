import pandas as pd
import os
from threading import Thread
import threading
from Sourcedata import data_use


class TrainingData:

    def __init__(self) -> None:
        self.SourceData = data_use.DataBase()
        # 計算句子出錯數量
        self.ErrorCount = 0
        self.GoodCount = 0

    def data_import(self):
        """檔案匯入"""
        currentdir = os.getcwd()
        file_loc = os.path.join(currentdir, "zjy1.xlsx")
        d = pd.read_excel(file_loc)
        d = d[['Eng', 'entity']]
        return d

    def run(self):
        """主要執行將將資料轉換為可訓練資料"""
        d = self.data_import()
        Vocal_sentence, Vocal_label = self.add_Vocab()
        print("單詞處理完畢")
        file_sentence , file_label = self.deal_file(d)
        sentence = (d['Eng'].tolist()+Vocal_sentence ) ; labels = (file_label+Vocal_label)
        print("句子與單詞合併完成")

        if len(sentence) == len(labels):
                df = pd.DataFrame(list(zip(sentence, labels)), columns=[
                    'text', 'entity'])
                df = df[df["entity"] != "no entity"];print("None are clear out!")
                df.to_excel("trans_result.xlsx")
                print("Trans DONE") ;
                print(f"Good:{self.GoodCount}")
                print(f"Bad:{self.ErrorCount}")

        else:
            raise ValueError(
                f"Sentense and labels total count does not equeal,sentence:{len(sentence)} ;labels:{len(labels)}"
                )

    def deal_file(self,data):
        """將檔案匯入進去"""
        d = data
        sentence = self.__trans_sentense(d['Eng'].tolist())
        label = self.__trans_labels(sentence)
        return sentence,label

    def __trans_sentense(self, l: list):
        """句子轉換"""
        import nltk
        # 確定是否有英文單詞庫
        nltk.download("punkt")
        res = []
        for i in l:

            tmp=(nltk.word_tokenize(i))

            res.append(tmp)

        return res

    def __trans_labels(self, l: list):
        """標籤轉換"""
        res = []
        for i in l:
            tmp = []
            # 將一整句拿去查詢是否有實體
            tmp.extend(self.__look(i))
            try:
                if len(set(tuple(k) for k in res)) == 1:
                    res.append("no entity")
                else:
                    res.append(tmp)
            except Exception as e:
                raise SystemError(f"資料有誤{e}")
        return res

    def __from_file_get_word_and_BIO(self):
        """從食譜資料拿取單辭以及BIO實體標示"""
        ing_csv_1 = pd.read_csv("Sourcedata/res/ingredients.csv")
        tag_csv_1 = pd.read_csv("Sourcedata/res/tags.csv")
        ing_csv_2 = pd.read_csv("Sourcedata/res/Uinque_ing.csv")
        tag_csv_2 = pd.read_csv("Sourcedata/res/Unique_tag.csv")
        # 將處理好的資料進行修改
        ing = [i for i in ing_csv_1['word']]
        ing_BIO = [eval(i) for i in ing_csv_1['mark'].tolist()]
        tag = [i for i in tag_csv_1['word']]
        tag_BIO = [eval(i) for i in tag_csv_1['mark'].tolist()]
        all_ing = ing_csv_2["word"].tolist()
        all_tag = tag_csv_2['word'].tolist()
        return ing,tag,ing_BIO,tag_BIO,all_ing,all_tag

    def __look(self, sentence):
        """實體標記中"""
        res = []
        ing,tag,ing_BIO,tag_BIO,all_ing,all_tag = self.__from_file_get_word_and_BIO()
        ing_split = [];tag_split = []
        for i in ing:
            if " " not in i:
                ing_split.append(i)
            else:
                ing_split.append(i.split(" "))
        for i in tag:
            try:
                if " " not in i:
                    ing_split.append(i)
                else:
                    ing_split.append(i.split(" "))
            except:
                # float object exception
                pass
        Nutrition = ["Calories", "Carbohydrates", "Protein", "Fat",
                     "Fiber", "Sugar", "Sodium", "Cholesterol", "Vitamins", "Minerals"]
        chronic_diseases = ["Diabetes", "Hypertension", "Obesity",
                            "Heart disease", "Stroke", "Chronic respiratory diseases", "Cancer"]
        allergens = [
            "Pollen", "Dust mites", "Mold",
            "Pet dander", "Peanuts", "Tree nuts",
            "Milk",
            "Eggs",
            "Wheat",
            "Soy",
            "Fish",
            "Shellfish","allergens","allergen"
        ]
        step = ["step", "steps", "STEPS", "STEP"]
        time = ["hour", "hours", "seconds", "second", "minute", "minutes"]
        allergen_L = [word.lower() for word in allergens]
        chronic_diseases_L = [word.lower() for word in chronic_diseases]
        Nutrition_L = [word.lower() for word in Nutrition]
        upordown = ["fewer", "lower", "more", "under", "uper", "without"]
        special_mark = ["[ClS]","[SEP]","[PAD]"]
        if len(sentence) == 1:
            if sentence in ing:
                res.append("B-ING")
            elif sentence in tag:
                res.append("B-TAG")
            elif sentence in Nutrition or sentence in Nutrition_L:
                res.append("B-NUT")
            elif sentence in step:
                res.append("B-STP")
            elif sentence in time:
                res.append("B-TME")
            elif sentence in upordown:
                res.append("B-UDO")
        for i in sentence:
                try:
                    if type(int(i)) == type(1):
                        res.append("num")
                    else:
                        res.append("O")
                except:
                     res.append("O")
        def has_sublist(main_list, sublist):
            # 将子列表转换为集合
            sublist_set = set(sublist)
            # 遍历主列表中的子列表
            for sublist in main_list:
                # 如果子列表的所有值都在主列表中，返回True
                if set(sublist) <= sublist_set:
                    return True
            # 如果没有找到匹配的子列表，返回False
            return False
        # 判別食譜之原料
        for i in range(len(ing_split)):
            # 確認句子裡有ing的實體
            if has_sublist(ing_split[i],sentence):
                for j in range(len(sentence)):
                    # 先判斷是否為過敏原單詞
                    if sentence[j] in allergens or sentence[j] in allergen_L:
                        res[j]="B-ALG"
                    # 後判斷是否為實體
                    else:
                        ok = 0 ; num_ing = len(ing_BIO[j])
                        for k in range(num_ing):
                            try:
                                if sentence[j+k] == ing_split[i][k]:
                                    ok += 1
                            except:
                                # 若發生 index out of range ，跳過
                                pass
                        if ok == (num_ing):
                            try:
                                for k in range(num_ing):
                                    res[j+k] = ing_BIO[i][k]
                                self.GoodCount += 1
                            except:
                                # 若發生 index out of range ，跳過
                                pass
                        else:
                            pass
            else:
                self.ErrorCount += 1

        # 判別食譜其標籤
        for i in range(len(tag_split)):
            # 確認句子裡有ing的實體
            if has_sublist(tag_split[i],sentence):
                for j in range(len(sentence)):
                    # 先判斷是否為過敏原單詞
                    if sentence[j] in chronic_diseases or sentence[j] in chronic_diseases_L:
                        res[j]="B-DIS"
                    # 後判斷是否為實體
                    else:
                        ok = 0 ; num_tag = len(tag_BIO[j])
                        for k in range(num_tag):
                            try:
                                if sentence[j+k] == tag_split[i][k]:
                                    ok += 1
                            except:
                                pass
                        if ok == num_ing:
                            for k in range(num_tag):
                                res[j+k] = tag_BIO[i][k]
                            self.GoodCount += 1
                        else:
                            pass

        for i in range(len(sentence)):
            if sentence[i] in all_ing:
                res[i] = "B-ING"
            if sentence[i] in all_tag:
                res[i] = "B-TAG"

        for i in range(len(sentence)):
            if sentence[i] in Nutrition or sentence[i] in Nutrition_L:
                res[i]="B-NUT"
            elif sentence[i] in step:
                res[i]="B-STP"
            elif sentence[i] in time:
                res[i]="B-TME"
            elif sentence[i] in upordown:
                res[i]="B-UDO"

        return res

    def add_Vocab(self):
        """添加單詞至訓練資料中"""
        ing,tag,ing_BIO,tag_BIO,all_ing,all_tag = self.__from_file_get_word_and_BIO()
        all_word = ing+tag ; all_BIO = ing_BIO+tag_BIO
        sentence = [i for i in all_word] ; label = [i for i in all_BIO]

        Nutrition = ["Calories", "Carbohydrates", "Protein", "Fat",
                     "Fiber", "Sugar", "Sodium", "Cholesterol", "Vitamins", "Minerals"]
        chronic_diseases = ["Diabetes", "Hypertension", "Obesity",
                            "Heart disease", "Stroke", "Chronic respiratory diseases", "Cancer"]
        allergens = [
            "Pollen", "Dust mites", "Mold",
            "Pet dander", "Peanuts", "Tree nuts",
            "Milk",
            "Eggs",
            "Wheat",
            "Soy",
            "Fish",
            "Shellfish",
            "Insect stings",
            "Bee venom",
            "Wasp venom",
            "Latex",
            "Medications",
            "Penicillin",
            "Aspirin",
            "Ibuprofen",
            "Cockroach droppings"
        ]
        step = ["step", "steps", "STEPS", "STEP"]
        time = ["hour", "hours", "seconds", "second", "minute", "minutes"]
        allergen_L = [word.lower() for word in allergens]
        chronic_diseases_L = [word.lower() for word in chronic_diseases]
        Nutrition_L = [word.lower() for word in Nutrition]
        upordown = ["fewer", "lower", "more", "under", "uper", "without"]
        all = [Nutrition, chronic_diseases, allergens,
               step, time, allergen_L, chronic_diseases_L, Nutrition_L, upordown]
        for i in all:
            for j in i:
                sentence.append(j)
                label.append([self.__look(j)])
        return sentence, label


if __name__ == "__main__":
    # data_use.DataBase().for_ing()
    # TrainingData.test()
    TrainingData().run()
