from statics.Statics import Statics
import pandas as pd
import numpy as np

self = Statics()

countries = self.get_all_countries()
valid_years = self.get_valid_years(end_year=2024)

country = np.random.choice(countries)
year = np.random.choice(valid_years)

def test_six_idx():
    assert 0 <= self.get_strong_point_num(year=year, country=country) <=3
    assert 0 <= self.get_hhi_index(2024) <= 1
    assert 0 <= self.get_award_rate(2024) <= 1
    # self.get_host(2024)
    assert 0 <= len(self.get_participates(2024)['Name'])
    assert 0 <= self.get_history_performance(2024, country='China')

def test_get_host():
    # year = 1908
    df1 = self.athlete.csv_file
    df1 = df1[['Team','NOC']]
    df1 = df1.drop_duplicates()
    # print(df1)
    full_name = self.host.csv_file[ self.host.csv_file['Year']==year ].iloc[0,1]
    
    if full_name in ['United Kingdom']:
        full_name = 'England'

    short_name = df1[ df1['Team'] == full_name ]    
    # print(short_name)
    assert len(short_name) == 1, "Test year is %d, full name is %s." % (year, full_name)
    print(short_name.iloc[0,1])
    

if __name__ == '__main__':
    test_six_idx()
    test_get_host()
    
    