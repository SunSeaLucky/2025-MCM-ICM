import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Preprocessor:
    def __init__(self, file_name, test_mode=True):
        self.dir = "E:\\Personal\\Contests\\2025-MCM-ICM\\public\\赛题\\2025_MCM-ICM_Problems\\2025_Problem_C_Data\\" + file_name
        self.encoding = "Windows-1252"
        self.test_mode = test_mode
        self.file_name = file_name
        
        self.csv_file = self.__read__()
        self.__preprocess_info__()
        
    def __read__(self):
        return pd.read_csv(self.dir, encoding=self.encoding)
    
    def __preprocess_info__(self):
        na_num = self.csv_file.isnull().sum().sum()
        print("%s 的空值数量：%d" % (self.file_name, na_num))
        
    def save_csv(self):
        self.csv_file.to_csv(self.file_name + "_preprocessed.csv", index=False, encoding="utf-8")

class Athlete(Preprocessor):
    def __init__(self, test_mode=True):
        super().__init__(file_name="summerOly_athletes.csv", test_mode=test_mode)

class Host(Preprocessor):
    def __init__(self, test_mode=True):
        super().__init__(file_name="summerOly_hosts.csv", test_mode=test_mode)
        
class Medal(Preprocessor):
    def __init__(self, test_mode=True):
        super().__init__(file_name="summerOly_medal_counts.csv", test_mode=test_mode)
        
class Program(Preprocessor):
    def __init__(self, test_mode=True):
        super().__init__(file_name="summerOly_programs.csv", test_mode=test_mode)
        self.__preprocess__()

    def __preprocess__(self):
        self.csv_file.loc[69, '1924'] = 0
        self.csv_file.loc[70, '1924'] = 0
        self.csv_file = self.csv_file.iloc[:-3, :]
        
        self.csv_file.loc[39, 'Code'] = "JDP"
        self.csv_file.loc[47, 'Code'] = "ROQ"

        self.csv_file.loc[37, 'Code'] = "IndoorHBL"
        self.csv_file.loc[38, 'Code'] = "FieldHBL"

        self.csv_file.loc[42, 'Code'] = "SixesLAX"
        self.csv_file.loc[43, 'Code'] = "FieldLAX"
        
        self.csv_file.loc[49, '1896'] = 0
        self.csv_file.loc[52, '1896'] = 0
       
        self.csv_file.fillna(0, inplace=True)
        self.csv_file.replace('•', 0, inplace=True)

        self.csv_file.drop(columns=['Sport', 'Discipline', 'Sports Governing Body'], inplace=True)