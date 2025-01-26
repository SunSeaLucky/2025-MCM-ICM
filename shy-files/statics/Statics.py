from preprocess.Preprocessor import Athlete
from preprocess.Preprocessor import Host
from preprocess.Preprocessor import Medal
from preprocess.Preprocessor import Program
from preprocess.Preprocessor import RawDataset
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import numpy as np

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
        
        if country is not None: 
            df = df[ (df['Medal'] != "No medal") & (df['Year']==year) & (df['NOC']==country) ]
            df = df.drop_duplicates(subset=['Sport', 'Event', 'Medal'])            
        else:            
            df = df[ (df['Medal'] != "No medal") & (df['Year']==year)]
            df = df.drop_duplicates(subset=['NOC', 'Sport', 'Event', 'Medal'])
        
        return df
    
    def get_gold_medal(self, year: int = 2024, country: str = None):
        df = self.athlete.csv_file
        
        if country:        
            df = df[ (df['Medal'] == "Gold") & (df['Year']==year) & (df['NOC']==country) ]
            df = df.drop_duplicates(subset=['Sport', 'Event', 'Medal'])
        else:
            df = df[ (df['Medal'] == "Gold") & (df['Year']==year)]
            df = df.drop_duplicates(subset=['NOC', 'Sport', 'Event', 'Medal'])
        
        return df
    
    def get_silver_medal(self, year: int = 2024, country: str = None):
        df = self.athlete.csv_file
        
        if country:        
            df = df[ (df['Medal'] == "Silver") & (df['Year']==year) & (df['NOC']==country) ]
            df = df.drop_duplicates(subset=['Sport', 'Event', 'Medal'])
        else:
            df = df[ (df['Medal'] == "Silver") & (df['Year']==year)]
            df = df.drop_duplicates(subset=['NOC', 'Sport', 'Event', 'Medal'])
        
        return df
    
    def get_bronze_medal(self, year: int = 2024, country: str = None):
        df = self.athlete.csv_file
        
        if country:        
            df = df[ (df['Medal'] == "Bronze") & (df['Year']==year) & (df['NOC']==country) ]
            df = df.drop_duplicates(subset=['Sport', 'Event', 'Medal'])
        else:
            df = df[ (df['Medal'] == "Bronze") & (df['Year']==year)]
            df = df.drop_duplicates(subset=['NOC', 'Sport', 'Event', 'Medal'])
        
        return df
    
    def get_all_countries(self):
        '''
        返回排序后的所有国家缩写
        '''
        return sorted(list(set(self.athlete.csv_file['NOC'])))
    
    def query_country(self, country:str = 'USA'):
        return self.get_all_countries().index(country)
    
    def query_idx(self, idx: int = 0) -> str:
        return self.get_all_countries()[idx]

    def get_country_never_awarded(self, year: int = 2024):
        '''
        计算 `year` 及之前，从未拿奖的国家，并升序输出。
        
        输出的国家包括未参赛的国家。
        '''
        all_countries = set(self.get_all_countries())
        
        df = self.athlete.csv_file
        df = df[ (df['Medal']!='No medal') & (df['Year'] <= year) ]
        
        awarded_countries = set(df['NOC'])
        return sorted(list(all_countries - awarded_countries))
    
    def get_country_awarded_by_year(self, year: int = 2024):
        '''
        计算 `year` 年拿奖的国家，升序输出。
        '''
        df = self.athlete.csv_file
        df = df[ (df['Medal']!='No medal') & (df['Year'] == year) ]
        return sorted(list(set(df['NOC'])))
    
    def get_valid_zero_one_countries(self, year):
        valid_years = self.get_valid_years()
        
        assert year >= valid_years[3], f"Invalid year {year}."

        award_countries = self.get_country_awarded_by_year(year=year)
        never_awarded_countries_by_year = self.get_country_never_awarded(year=valid_years[valid_years.index(year)-1])
        zero_one_countries = set(award_countries) & set(never_awarded_countries_by_year)

        df = self.athlete.csv_file

        year_ip1_countries = df[df['Year'] == valid_years[valid_years.index(year)-1] ]['NOC'].unique()
        year_ip2_countries = df[df['Year'] == valid_years[valid_years.index(year)-2] ]['NOC'].unique()
        year_ip3_countries = df[df['Year'] == valid_years[valid_years.index(year)-3] ]['NOC'].unique()

        to_remove = set()
        for i in zero_one_countries:
            if i in year_ip1_countries and i in year_ip2_countries and i in year_ip3_countries:
                continue
            else:
                to_remove.add(i)            
        return zero_one_countries - to_remove
    
    def get_valid_zero_zero_countries(self, year):
        valid_years = self.get_valid_years()
        
        assert year >= valid_years[3], f"Invalid year {year}."

        # award_countries = s.get_country_awarded_by_year(year=year)
        never_awarded_countries_by_year = set()
        # zero_one_countries = set(award_countries) & set(never_awarded_countries_by_year)

        df = s.athlete.csv_file

        year_ip1_countries = df[df['Year'] == valid_years[valid_years.index(year)-1] ]['NOC'].unique()
        year_ip2_countries = df[df['Year'] == valid_years[valid_years.index(year)-2] ]['NOC'].unique()
        year_ip3_countries = df[df['Year'] == valid_years[valid_years.index(year)-3] ]['NOC'].unique()

        to_remove = set()
        for i in never_awarded_countries_by_year:
            if i in year_ip1_countries and i in year_ip2_countries and i in year_ip3_countries:
                continue
            else:
                to_remove.add(i)            
        return never_awarded_countries_by_year - to_remove
    
    def get_zero_one_idx(self, year, country):
        df = self.athlete.csv_file
        valid_years = self.get_valid_years()

        sport = df.groupby(['Sport'])\
                    .size()\
                    .reset_index(name='Participants')\
                    .sort_values(by='Participants')['Sport']
        rare_sport = sport[ int(len(sport) * 0.05):int(len(sport) * 0.5) ]
        
        # Participants Growth Rate
        i = valid_years.index(year)
        s1 = set(df[ (df['Year']==valid_years[i-1] ) & (df['NOC']==country) ]['Name'])
        s2 = set(df[ (df['Year']==valid_years[i-2] ) & (df['NOC']==country) ]['Name'])
        s3 = set(df[ (df['Year']==valid_years[i-3] ) & (df['NOC']==country) ]['Name'])
        
        a1 = len(s1)
        a2 = len(s2)
        a3 = len(s3)
        pgr = (max(0, a2 - a3) + max(0, a2 - a1)) / 2   
        
        # New Project Index
        event_three_years = df[ (df['Year']==valid_years[i-1]) | (df['Year']==valid_years[i-2]) | (df['Year']==valid_years[i-3]) ]['Sport']
        event_one_to_i_plus_four_years = df[ df['Year'] < valid_years[i-3] ]['Sport']
        npi = len(set(event_three_years) - set(event_one_to_i_plus_four_years))
        
        # LPIR
        k1 = len(set(s1) & set(rare_sport))
        k2 = len(set(s2) & set(rare_sport))
        k3 = len(set(s3) & set(rare_sport))
        lpir = (max(0, k2-k1) + max(0, k3-k2))/2    
        
        return pgr, npi, lpir
    
    def host_effect_check(self):
        valid_years = self.get_valid_years()

        mth = []
        mt = []
        d = []

        for i in range(len(valid_years)-2):
            y1 = valid_years[i]
            y2 = valid_years[i+1]
            y3 = valid_years[i+2]
            
            host = self.get_host(y2)
            
            a1 = self.get_total_medal(year=y1, country=host).shape[0]
            a2 = self.get_total_medal(year=y2, country=host).shape[0]
            a3 = self.get_total_medal(year=y3, country=host).shape[0]

            mth.append( (a1+a3) / 2.0)
            mt.append(a2)
            d.append(abs(mth[len(mth)-1] - mt[len(mt)-1]))
            
        bar_d = np.mean(d)

        len_d = len(d)

        t = bar_d / (np.std(d) / np.sqrt(len_d))

        alpha = 0.05  # 显著性水平

        critical_value = stats.t.ppf(1 - alpha/2, len_d-1)

        p_value = 2 * (1 - stats.t.cdf(abs(t), len_d-1))

        print("t: %f" % t)
        print("p-value: %f" % p_value)
        print("critical value: %f" % critical_value)
        print("假设检验结果：%s" % str(t >= critical_value))