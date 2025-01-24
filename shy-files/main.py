from statics.Statics import Statics
import pandas as pd

self = Statics()

# print(self.get_strong_point_num(year=2024, country='China'))
# print(self.get_hhi_index(2024))
# print(self.get_award_rate(2024))
# print(self.get_host(2024))
# print(self.get_participate_num(2024))
# print(self.history_performance(2024, country='China'))

countries = self.get_all_countries()
valid_years = self.get_valid_years()

dataset = []

cnt = 0

for year in valid_years:
    for country in countries:
        print("Processing: %d year, %s country" % (year, country))
        cnt += 1
        if cnt >= 15:
            dataset = pd.DataFrame(data=dataset, columns=['Year','NOC', 'strong_point', 'hhi', 'award_rate', 'host', 'participate_num', 'history_performance'])
            dataset.to_csv('statics.csv', index=False)
            exit()
        arr = []
        arr.append(year)
        arr.append(country)
        arr.append(self.get_strong_point_num(year=year, country=country))
        arr.append(self.get_hhi_index(year))
        arr.append(self.get_award_rate(year))
        arr.append(self.get_host(year))
        arr.append(self.get_participate_num(year))
        arr.append(self.get_history_performance(year, country=country))
        dataset.append(arr)

# print(countries)
# print(valid_years)




        