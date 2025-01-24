import sys  
sys.path.append('../')

from preprocess.Preprocessor import Athlete
from preprocess.Preprocessor import Host
from preprocess.Preprocessor import Medal
from preprocess.Preprocessor import Program
import matplotlib.pyplot as plt

class Statics:
    def __init__(self):
        self.athlete = Athlete()
        self.host = Host()
        self.medal = Medal()
        self.program = Program()
        
    def get_hhl_index(self, year: int = 2024):
        '''
        按照年份计算 HHL
        '''
        df = self.athlete.csv_file

        # hhl_df = df.query('Medal != "No medal" and Year == %d' % year)
        hhl_df = df[ (df['Medal']!="No medal") & (df['Year']==year) ]
        hhl_df = hhl_df.drop_duplicates(subset=['NOC', 'Sport', 'Event', 'Medal'])
        hhl_count = hhl_df.groupby(['Sport']).size().reset_index(name='MedalCount')

        # 计算总奖牌数  
        total_medals = hhl_count["MedalCount"].sum()  

        # 计算每个项目的份额  
        hhl_count["Share"] = hhl_count["MedalCount"] / total_medals  

        # 计算 HHL 值  
        return (hhl_count["Share"] ** 2).sum()
    
    def get_hhl_index_by_country(self, year: int = 2024, country: str = "USA"):
        '''
        按照年份和国家计算 HHL
        '''
        df = self.athlete.csv_file
        
        hhl_df = df[ (df['Medal']!="No medal") & (df['Year']==year) & (df['NOC']==country) ]
        hhl_df = hhl_df.drop_duplicates(subset=['NOC', 'Sport', 'Event', 'Medal'])
        hhl_count = hhl_df.groupby(['Sport']).size().reset_index(name='MedalCount')

        # 计算总奖牌数  
        total_medals = hhl_count["MedalCount"].sum()  

        # 计算每个项目的份额  
        hhl_count["Share"] = hhl_count["MedalCount"] / total_medals  

        # 计算 HHL 值  
        return (hhl_count["Share"] ** 2).sum()
    
    def draw_hhl_index(self,start_year: int = 1950, end_year: int = 2024):
        k = set(self.athlete.csv_file['Year'])

        k = k.intersection(range(start_year, end_year+1))
                
        X = []
        y = []

        k = sorted(list(k))

        for i in k:
            X.append(i)
            y.append(self.get_hhl_index(i))

        plt.xlabel('Year')
        plt.ylabel('HHL Index')
        plt.plot(X, y)
    
    def get_event_relative(self, year: int = 2024, head: int = 3):
        '''
        获取项目相关性。计算国家在各项目的历史奖牌占比。默认计算 2024 年的前 3 项的运动的相对比赛次数。
        '''
        df = self.athlete.csv_file
        medal_df = df.query('Medal != "No medal" and Year == @year')
        medal_df = medal_df.drop_duplicates(subset=['NOC', 'Sport', 'Event', 'Medal'])

        # 统计每个国家每个运动的奖牌数
        medal_count = medal_df.groupby(['NOC', 'Sport']).size().reset_index(name='MedalCount')
        
        # 计算每个国家的总奖牌数  
        total_medals = medal_count.groupby('NOC')['MedalCount'].sum().reset_index(name='TotalMedals')  
        total_medals.sort_values(by='TotalMedals', ascending=False)  
        
        # 合并总奖牌数到每个运动的奖牌数  
        medal_count = medal_count.merge(total_medals, on='NOC')  
        medal_count.sort_values(by=['TotalMedals', 'MedalCount'], ascending=False, inplace=True)
        
        # 计算每个运动的奖牌占比  
        medal_count['Rate'] = medal_count['MedalCount'] / medal_count['TotalMedals']  
        
        return medal_count.groupby('NOC').head(head) 