#  --coding:utf8--**
import pandas as pd
# from recipe.models import Recipe_At , Recipe_Ob
import os
class DataBase:
    def __init__(self) -> None:
        """在物件生成時，做什麼事情"""
        pass

    def resource(self):
        """
        資料預處理後結果
        """
        current = os.getcwd()
        try:
            # 依照本地進行設計
            d1 = pd.read_csv(os.path.join(current,"data","RAW_recipes.csv"))
            d2 = pd.read_csv(os.path.join(current,"data","Food Ingredients and Recipe Dataset with Image Name Mapping.csv"))
        except FileNotFoundError:
            """only跑該檔案"""
            try:
                d1 = pd.read_csv(os.path.join(current,"SourceData/RAW_recipes.csv"))
                d2 = pd.read_csv(os.path.join(current,"SourceData/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"))
            except FileNotFoundError:
                r = os.path.join(current,"recipe/data")
                raise FileNotFoundError(f"請確認是否有食譜檔案在資料夾中，路徑為下{r}")

        return d1 , d2

    def info(self):
        """
        Food Ingredients and Recipe Dataset with Image Name Mapping.csv
        """
        data = pd.read_csv("./Food Ingredients and Recipe Dataset with Image Name Mapping.csv")
        print("-"*50)
        print("資料表:Food Ingredients and Recipe Dataset with Image Name Mapping.csv")
        print(f"本資料集大小為{data.shape}")
        print(f"其欄位:{data.columns.tolist()}")
        print("第一筆資料如下")
        print(data.head(1))

        """
        表格結構：
        ----------
        Unnamed: 0： 可能是索引列，不包含實際的資訊。
        Title： 食譜標題。
        Ingredients： 食材清單。
        Instructions： 食譜製作步驟。
        Image_Name： 食譜相關圖片的檔名。
        Cleaned_Ingredients： 已經清理過的食材資訊。
        資料內容：
        ----------
        Title 和 Image_Name： 提供了食譜的名稱和相應的圖片檔名。
        Ingredients 和 Cleaned_Ingredients： 分別提供了原始和清理過的食材資訊。
        Instructions： 包含了製作食譜的步驟。
        使用性質：
        ----------
        可以通過標題快速查找特定食譜。
        食材和步驟的資訊可以用於系統的呈現和查詢功能。
        潛在改進：
        ----------
        可能需要進一步的資料清理，確保資訊的一致性和完整性。
        如果需要支援多語言，可能需要擴充語言支援。
        """
        print("-"*50)

        print("資料表:Food Ingredients and Recipe Dataset with Image Name Mapping.csv")
        data1 = pd.read_csv("/Users/apple/Desktop/Code/Project/KBQA_Bert__RecipeRecommendations/backend/Meibuy/backend/recipe/data/RAW_recipes.csv")
        print(f"本資料集大小為{data1.shape}")
        print(f"其欄位:{data1.columns.tolist()}")
        print("第一筆資料如下")
        print(data1.head(1))

        print("-"*50)

    def know_value(self):
        """
        得取欄位的值
        tags、ingredients、minutes

        Processing Threads: 100%|██████████| 3/3 [02:27<00:00, 49.15s/it]
        為處理時間!
        """
        from threading import Thread ; from tqdm import tqdm
        d1,d2 = self.resource() ; data = d1
        t1 = Thread(target=self.find_unqiue_value,args=("tags",data,))
        t2 = Thread(target=self.find_unqiue_value,args=("ingredients",data,))
        # t3 = Thread(target=self.find_unqiue_value,args=("minutes",))
        threads = [t1, t2]
        # 使用 tqdm 顯示進度條
        with tqdm(total=len(threads), desc="Processing Threads") as pbar:
            for t in threads:
                t.start()
            for t in threads:
                t.join()
                pbar.update(1)
        print("Done")

    def find_unqiue_value(self,key,data):
        keys = {"tags":1,"ingredients":2}
        word = [] ; BIO_tags = []
        # 將資料儲存於陣列
        for index, row in data.iterrows():
            for i in eval(row[key]):
                word.append(i)
        # 找出唯一值
        word = set(word) ; words = list(word)
        # 將唯一的單詞轉為BIO
        for word in words:
            tmp_BIO = self.__mark_data(
                what = keys[key],
                word = word
                )
            BIO_tags.append(tmp_BIO)
        # 儲存
        df = pd.DataFrame({
            "word":words,
            "mark":BIO_tags
            })
        df.to_csv(f"res/{key}.csv")



    def __mark_data(self,what,word):
        """
        專案需求為Bert實體標記，透過BIO的架構去做辨識
        mark_data 其標記資料的BIO
        """
        word = word.split(" ")
        num = len(word)
        # 選擇標籤
        if what == 1 :
            mark = ["B-TAG","I-TAG"]
        elif what == 2:
            mark = ["B-ING","I-ING"]
        else:
            raise KeyError("未知參數，只有tags、ingredients")
        if num == 1 :
            return [mark[0]]
        else:
            l = [mark[0]]
            for i in range(1,num):
                l.append(mark[1])
            return l

    def getSingleWord(self):
        d1,d2 = self.resource()
        word = [eval(i) for i in d1['ingredients'].tolist() ] ; all_word = [] ; word_BIO = []
        tag =  [eval(i) for i in  d1['tags'].tolist() ] ; all_tag = [] ; tag_BIO = []
        for i in word:
            for j in i:
                for k in j.split(" "):
                    all_word.append(k)
        for i in tag:
            for j in i:
                for k in j.split(" "):
                    all_tag.append(k)
        word = pd.DataFrame({"word":all_word}) ; tag = pd.DataFrame({"word":all_tag})
        res1 = word.value_counts(ascending=False)[0:1500]
        res2 = tag.value_counts(ascending=False)[0:300]
        res1.to_csv("res/Uinque_ing.csv") ; res2.to_csv("res/Unique_tag.csv")








class Trans_db(DataBase):
    """將原資料進行轉換"""
    def trans(self):
        """將原資料庫資料進行轉換!"""
        d1 , d2 = super().resource()
        data = d1
        # print(d1)
        from threading import Thread ;

        t1 = Thread(target=self.__toObject,args=(data,))
        t2 = Thread(target=self.__toAttribute,args=(data,))
        t1.start() ; t2.start()
        print("處理結束已將資料存於資料庫")

    def __toObject(self, data):
        for index, row in data.iterrows():
            Recipe_Ob.objects.create(
                rid=row['id'],
                name=row['name'],
                tags=row['tags'],
                steps=row['steps'],
                description=row['description'],
                ingredients=row['ingredients']
            ).save()

        print('O_complete')

    def __toAttribute(self, data):
        for index, row in data.iterrows():
            Recipe_At.objects.create(
                rid=row['id'],
                minutes=row['minutes'],
                nutrition=row['nutrition'],
                n_steps=row['n_steps'],
                n_ingredients=row['n_ingredients']
            ).save()
        print("A_complete")

    def translate_to_chinese(text):
        from googletrans import Translator
        translator = Translator()
        translation = translator.translate(text, dest='zh-tw')
        return translation.text


if __name__=="__main__":
    # DataBase().know_value()
    DataBase().getSingleWord()