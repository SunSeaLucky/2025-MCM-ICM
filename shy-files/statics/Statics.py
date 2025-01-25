from preprocess.Preprocessor import Athlete
from preprocess.Preprocessor import Host
from preprocess.Preprocessor import Medal
from preprocess.Preprocessor import Program
from preprocess.Preprocessor import RawDataset
import matplotlib.pyplot as plt
import pandas as pd

class Statics:
    def __init__(self):
        self.athlete = Athlete()
        self.host = Host()
        self.medal = Medal()
        self.program = Program()
        self.raw_dataset = RawDataset()
        self.matlab_dir = './matlab-plotter/'
        
    def get_valid_years(self, start_year: int = 1896, end_year: int = 2024):
        '''
        时间是闭区间。
        '''
        k = set(self.athlete.csv_file['Year'])
        k = k.intersection(range(start_year, end_year+1))        
        return sorted(list(k))
        
    def get_strong_point_num(self, year: int = 2024, country: str='USA'):
        '''
        六个指标中的第一个，参赛项目。
        '''
        valid_years = self.get_valid_years()
        assert year in valid_years, "Invalid year."

        # df = self.program.csv_file
        # cur_year_events = set(df['Sport'] * df[str(year) + '*' if year==1906 else str(year) ].apply(lambda x: 1 if x > 0 else 0))

        df = self.athlete.csv_file
        df = df[df['Year']==year]
        
        cur_year_events = set(df['Sport'])
        strong_points = set(self.get_event_relative_by_country(year=year, country=country, head=3)['Sport'])

        assert len(strong_points) <= 3, "Most 3 strong points."
        return len(strong_points.intersection(cur_year_events))     
        
    def get_hhi_index(self, year: int = 2024):
        '''
        按照年份计算 hhi
        '''
        df = self.athlete.csv_file

        # hhi_df = df.query('Medal != "No medal" and Year == %d' % year)
        hhi_df = df[ (df['Medal']!="No medal") & (df['Year']==year) ]
        hhi_df = hhi_df.drop_duplicates(subset=['NOC', 'Sport', 'Event', 'Medal'])
        hhi_count = hhi_df.groupby(['Sport']).size().reset_index(name='MedalCount')

        # 计算总奖牌数  
        total_medals = hhi_count["MedalCount"].sum()  

        # 计算每个项目的份额  
        hhi_count["Share"] = hhi_count["MedalCount"] / total_medals  

        # 计算 hhi 值  
        return (hhi_count["Share"] ** 2).sum()
    
    def get_hhi_index_by_country(self, year: int = 2024, country: str = "USA"):
        '''
        按照年份和国家计算 HHI
        '''
        df = self.athlete.csv_file
        
        hhi_df = df[ (df['Medal']!="No medal") & (df['Year']==year) & (df['NOC']==country) ]
        hhi_df = hhi_df.drop_duplicates(subset=['NOC', 'Sport', 'Event', 'Medal'])
        hhi_count = hhi_df.groupby(['Sport']).size().reset_index(name='MedalCount')

        # 计算总奖牌数  
        total_medals = hhi_count["MedalCount"].sum()  

        # 计算每个项目的份额  
        hhi_count["Share"] = hhi_count["MedalCount"] / total_medals  

        # 计算 hhi 值  
        return (hhi_count["Share"] ** 2).sum()
    
    def draw_hhi_index(self,start_year: int = 1896, end_year: int = 2024):
        '''
        绘制 HHI 指数变化图，默认从 1896 年到 2024 年
        '''
        k = self.get_valid_years(start_year, end_year)
               
        X = []
        y = []

        for i in k:
            X.append(i)
            y.append(self.get_hhi_index(i))

        plt.xlabel('Year')
        plt.ylabel('HHI Index')
        plt.title('HHI Index Change')
        plt.plot(X, y)
        
        data = pd.DataFrame({"Year": X, "HHIndex": y})
        data.to_csv(self.matlab_dir + 'data_hhi.csv', index=False)
    
    def get_event_relative_by_country(self, year: int = 2024, country: str = 'USA', head: int = 3):
        '''
        获取项目相关性。计算国家在各项目的历史奖牌占比。默认计算 2024 年的前 3 项的运动的相对比赛次数。
        '''
        df = self.athlete.csv_file
        # medal_df = df.query('Medal != "No medal" and Year == @year')
        medal_df = df[ (df['Medal']!="No medal") & (df['Year'] <= year) & (df['NOC']==country) ]
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
    
    def get_event_relative(self, year: int = 2024, head: int = 3):
        '''
        获取项目相关性。计算国家在各项目的历史奖牌占比。默认计算 2024 年的前 3 项的运动的相对比赛次数。
        '''
        df = self.athlete.csv_file
        # medal_df = df.query('Medal != "No medal" and Year == @year')
        medal_df = df[ (df['Medal']!="No medal") & (df['Year']==year) ]
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

    def get_award_rate(self, year: int = 2024, country: str = 'USA'):
        awarded = self.get_total_medal(year=year, country=country)
        all = self.get_participates(year=year, country=country)
        
        awarded = float(len(set(awarded['Name'])))
        all = float(len(set(all['Name'])))

        assert awarded <= all
        
        return 0 if all == 0 else awarded / all
    
    def get_host(self, year: int = 2024):
        df1 = self.athlete.csv_file
        df1 = df1[['Team','NOC']]
        df1 = df1.drop_duplicates()
        full_name = self.host.csv_file[ self.host.csv_file['Year']==year ].iloc[0,1]
        
        if full_name in ['United Kingdom']:
            full_name = 'England'

        short_name = df1[ df1['Team'] == full_name ]    
        
        assert len(short_name) == 1, "Test year is %d, full name is %s." % (year, full_name)

        return short_name.iloc[0,1]
    
    def get_history_performance(self, year: int = 2024, country:str = 'USA'):
        years = self.get_valid_years(end_year=year)[-3:]
        cnt = len(years)
        sum = 0
        
        for y in years:
            sum += self.get_total_medal(year=y, country=country).shape[0]
            
        return float(sum) / cnt

    def get_participates(self, year: int = 2024, country: str = None):
        df = self.athlete.csv_file
        if country is not None:
            df = df[ (df['Year']==year) & (df['NOC']==country) ]
        else:            
            df = df[ (df['Year']==year) ]
        return df

    def get_total_medal(self, year: int = 2024, country: str = None):
        df = self.athlete.csv_file
        df = df.drop_duplicates(subset=['NOC', 'Sport', 'Event', 'Medal'])
        
        if country:        
            df = df[ (df['Medal'] != "No medal") & (df['Year']==year) & (df['NOC']==country) ]
        else:
            df = df[ (df['Medal'] != "No medal") & (df['Year']==year)]
        
        return df
    
    def get_gold_medal(self, year: int = 2024, country: str = None):
        df = self.athlete.csv_file
        df = df.drop_duplicates(subset=['NOC', 'Sport', 'Event', 'Medal'])
        
        if country:        
            df = df[ (df['Medal'] == "Gold") & (df['Year']==year) & (df['NOC']==country) ]
        else:
            df = df[ (df['Medal'] == "Gold") & (df['Year']==year)]
        
        return df
    
    def get_silver_medal(self, year: int = 2024, country: str = None):
        df = self.athlete.csv_file
        df = df.drop_duplicates(subset=['NOC', 'Sport', 'Event', 'Medal'])
        
        if country:        
            df = df[ (df['Medal'] == "Silver") & (df['Year']==year) & (df['NOC']==country) ]
        else:
            df = df[ (df['Medal'] =="Silver") & (df['Year']==year)]
        
        return df
    
    def get_bronze_medal(self, year: int = 2024, country: str = None):
        df = self.athlete.csv_file
        df = df.drop_duplicates(subset=['NOC', 'Sport', 'Event', 'Medal'])
        
        if country:        
            df = df[ (df['Medal'] == "Bronze") & (df['Year']==year) & (df['NOC']==country) ]
        else:
            df = df[ (df['Medal'] == "Bronze") & (df['Year']==year)]
        
        return df
    
    def get_all_countries(self):
        return sorted(list(set(self.athlete.csv_file['NOC'])))